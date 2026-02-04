import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import time
import json
import os
import copy
import random
import datetime
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.utils import shuffle
import sys

# Warmup + step LR scheduler
class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, step_size, gamma=0.1, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        step_epoch = self.last_epoch - self.warmup_epochs
        decay_factor = self.gamma ** (step_epoch // max(1, self.step_size))
        return [max(base_lr * decay_factor, self.min_lr) for base_lr in self.base_lrs]


# Logging configuration
def setup_logging(log_file='log.txt'):
    """
    Setup logging configuration
    """
    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Create log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Performance monitoring decorator
def log_performance(func_name):
    """Performance monitoring decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.info(f"[Performance] {func_name} execution time: {end_time - start_time:.2f}s")
            return result
        return wrapper
    return decorator

# Random seed setting
def set_random_seed(seed=42):
    """
    Set all random seeds for reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")

# Model initialization helper
def apply_sparse_init(m):
    """Apply sparse initialization to model layers"""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ROC and threshold computation
def optimal_thresh(fpr, tpr, thresholds, p=0):
    """Calculate optimal threshold"""
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    """Calculate multi-label ROC curves"""
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
        
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        
        try:
            c_auc = roc_auc_score(label, prediction)
            logging.info(f"Class {c} ROC AUC score: {c_auc:.4f}")
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                logging.warning(f"Class {c} has only one class present, ROC AUC set to 1.0")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
        
    return aucs, thresholds, thresholds_optimal

# Model checkpoint management
def save_model_checkpoint(args, fold, save_path, model, epoch, accuracy, auc, thresholds_optimal, model_type='best', additional_metrics=None):
    """Save model checkpoint with training metrics"""
    save_name = os.path.join(save_path, f'fold_{fold}_{model_type}.pth')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'auc': auc,
        'fold': fold,
        'model_type': model_type,
        'thresholds_optimal': thresholds_optimal
    }
    
    # Add additional metrics if provided
    if additional_metrics:
        checkpoint.update(additional_metrics)
    
    torch.save(checkpoint, save_name)
    
    # Save thresholds separately
    thresh_file = os.path.join(save_path, f'fold_{fold}_{model_type}.json')
    with open(thresh_file, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)
    
    model_size = os.path.getsize(save_name) / (1024 * 1024)  # MB
    logging.info(f'Model saved: {save_name} (type: {model_type}, size: {model_size:.2f}MB)')
    logging.info(f'Model metrics: Epoch={epoch}, Accuracy={accuracy:.4f}, AUC={auc}')
    logging.info(f'Optimal thresholds: {[float(x) for x in thresholds_optimal]}')
    
    return save_name

def load_checkpoint(resume_path, fold, model_type='last'):
    """Load model checkpoint"""
    checkpoint_path = os.path.join(resume_path, f'fold_{fold}_{model_type}.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logging.info(f"Loaded checkpoint: {checkpoint_path}")
        logging.info(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
        logging.info(f"   - Accuracy: {checkpoint.get('accuracy', 'unknown'):.4f}")
        logging.info(f"   - AUC: {checkpoint.get('auc', 'unknown')}")
        return checkpoint
    else:
        logging.warning(f"Checkpoint not found for fold {fold}: {checkpoint_path}")
        return None

# Score calculation
def get_current_score(avg_score, aucs):
    """Calculate current comprehensive score"""
    if isinstance(aucs, list):
        return (sum(aucs) + avg_score) / 2
    else:
        return (aucs + avg_score) / 2

# Print utilities
def print_epoch_info(epoch, total_epochs, train_loss, test_loss, avg_score, aucs, dataset_name=''):
    """Print epoch information"""
    if isinstance(aucs, list) and len(aucs) > 1:
        if dataset_name.startswith('TCGA-lung'):
            info_msg = f'Epoch [{epoch}/{total_epochs}] train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}, avg_score: {avg_score:.4f}, auc_LUAD: {aucs[0]:.4f}, auc_LUSC: {aucs[1]:.4f}'
        else:
            auc_str = '|'.join('class-{}>>{:.4f}'.format(i, auc) for i, auc in enumerate(aucs))
            info_msg = f'Epoch [{epoch}/{total_epochs}] train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}, avg_score: {avg_score:.4f}, AUC: {auc_str}'
    else:
        auc_val = aucs[0] if isinstance(aucs, list) else aucs
        info_msg = f'Epoch [{epoch}/{total_epochs}] train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}, avg_score: {avg_score:.4f}, auc: {auc_val:.4f}'
    
    logging.info(f'[Training] {info_msg}')

def print_final_results(results, training_time, dataset_name=''):
    """Print final training results"""
    if not results:
        logging.warning("No results to display - all folds may have been completed or errors occurred")
        return
        
    accuracies = [r[0] for r in results]
    aucs_list = [r[1] for r in results]
    
    mean_ac = np.mean(accuracies)
    std_ac = np.std(accuracies)
    
    logging.info("=" * 80)
    logging.info("FINAL RESULTS")
    logging.info("=" * 80)
    logging.info(f"Average accuracy: {mean_ac:.4f} ± {std_ac:.4f}")
    
    if isinstance(aucs_list[0], list):
        # Multi-class case
        mean_auc = np.mean(aucs_list, axis=0)
        std_auc = np.std(aucs_list, axis=0)
        for i, (mean_score, std_score) in enumerate(zip(mean_auc, std_auc)):
            logging.info(f"Class {i} average AUC: {mean_score:.4f} ± {std_score:.4f}")
    else:
        # Single class case
        mean_auc = np.mean(aucs_list)
        std_auc = np.std(aucs_list)
        logging.info(f"Average AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    logging.info(f"Total training time: {training_time:.2f}s ({training_time/3600:.2f}h)")
    logging.info(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# TensorBoard integration utilities
def init_tensorboard(log_dir, run_name=None):
    """Initialize TensorBoard for experiment tracking"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        import os
        
        # Create log directory if it doesn't exist
        if run_name:
            log_path = os.path.join(log_dir, run_name)
        else:
            log_path = log_dir
        
        os.makedirs(log_path, exist_ok=True)
        
        writer = SummaryWriter(log_dir=log_path)
        logging.info(f"TensorBoard initialized with log directory: {log_path}")
        return writer
    except ImportError:
        logging.warning("TensorBoard not installed. Install with: pip install tensorboard")
        return None
    except Exception as e:
        logging.warning(f"Failed to initialize TensorBoard: {e}")
        return None

def log_tensorboard_metrics(tensorboard_writer, metrics, step=None):
    """Log metrics to TensorBoard"""
    if tensorboard_writer is not None:
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tensorboard_writer.add_scalar(key, value, step)
                elif isinstance(value, dict):
                    # Handle nested metrics
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            tensorboard_writer.add_scalar(f"{key}/{sub_key}", sub_value, step)
        except Exception as e:
            logging.warning(f"Failed to log to TensorBoard: {e}")

def log_tensorboard_model(tensorboard_writer, model, input_tensor, model_name="model"):
    """Log model graph to TensorBoard"""
    if tensorboard_writer is not None:
        try:
            tensorboard_writer.add_graph(model, input_tensor)
            logging.info(f"Model graph logged to TensorBoard: {model_name}")
        except Exception as e:
            logging.warning(f"Failed to log model graph to TensorBoard: {e}")

def finish_tensorboard(tensorboard_writer):
    """Close TensorBoard writer"""
    if tensorboard_writer is not None:
        try:
            tensorboard_writer.close()
            logging.info("TensorBoard writer closed")
        except Exception as e:
            logging.warning(f"Failed to close TensorBoard writer: {e}")

# Model parameter configuration utilities
def parse_model_params(args):
    """Parse model-specific parameters from command line arguments"""
    model_name = args.model
    params = {}
    
    if model_name == 'efficientmil_gru':
        params.update({
            'gru_hidden_size': args.gru_hidden_size,
            'gru_num_layers': args.gru_num_layers,
            'bidirectional': args.gru_bidirectional == 'True',
            'selection_strategy': args.gru_selection_strategy,
            'big_lambda': args.big_lambda,
            'dropout': args.dropout,
        })
    elif model_name == 'efficientmil_lstm':
        params.update({
            'lstm_hidden_size': args.lstm_hidden_size,
            'lstm_num_layers': args.lstm_num_layers,
            'bidirectional': args.lstm_bidirectional == 'True',
            'selection_strategy': args.lstm_selection_strategy,
            'big_lambda': args.big_lambda,
            'dropout': args.dropout,
        })
    elif model_name == 'efficientmil_mamba':
        params.update({
            'mamba_depth': args.mamba_depth,
            'd_state': args.mamba_d_state,
            'd_conv': args.mamba_d_conv,
            'expand': args.mamba_expand,
            'selection_strategy': args.mamba_selection_strategy,
            'big_lambda': args.big_lambda,
            'dropout': args.dropout,
        })
    
    return params

def log_model_info(model, model_name, params=None):
    """Log detailed model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model architecture: {model_name}")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Estimated model size: {total_params * 4 / (1024*1024):.2f} MB")
    
    if params:
        logging.info("Model specific parameters:")
        for key, value in params.items():
            logging.info(f"   {key}: {value}")

