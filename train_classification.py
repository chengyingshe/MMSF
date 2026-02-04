import argparse
import os
import logging
from typing import Optional, Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import multiprocessing as mp
from sklearn.metrics import roc_auc_score

from utils import setup_logging, set_random_seed, WarmupStepLR
from data import ClassificationDataset, GraphClassificationDataset, collate_fn, collate_fn_with_graph
from model import Network, initialize_weights
from clinical_encoders import build_clinical_embedding_loss_fn

def compute_l2_penalty(model, args, current_epoch=0, total_epochs=30):
    """Compute L2 regularization penalty for model parameters."""
    # Calculate current L2 lambda based on schedule
    if args.l2_decay_schedule == 'increasing':
        progress = current_epoch / total_epochs
        current_l2_lambda = min(args.l2_lambda + progress * (args.max_l2_lambda - args.l2_lambda), args.max_l2_lambda)
    elif args.l2_decay_schedule == 'decreasing':
        progress = current_epoch / total_epochs
        current_l2_lambda = max(args.l2_lambda - progress * (args.l2_lambda - args.max_l2_lambda * 0.1), args.max_l2_lambda * 0.1)
    else:  # constant
        current_l2_lambda = args.l2_lambda
    
    l2_penalty = 0.0
    
    if args.l2_penalty_type == 'all':
        # Apply L2 to all parameters
        for param in model.parameters():
            if param.requires_grad:
                l2_penalty += torch.norm(param, p=2) ** 2
    elif args.l2_penalty_type == 'classifier_only':
        # Apply L2 only to classifier parameters
        for name, param in model.named_parameters():
            if param.requires_grad and ('classifier' in name.lower() or 'fc' in name.lower() or 'linear' in name.lower()):
                l2_penalty += torch.norm(param, p=2) ** 2
    elif args.l2_penalty_type == 'attention_only':
        # Apply L2 only to attention/selection parameters
        for name, param in model.named_parameters():
            if param.requires_grad and ('attention' in name.lower() or 'selector' in name.lower() or 'gate' in name.lower()):
                l2_penalty += torch.norm(param, p=2) ** 2

    return current_l2_lambda * l2_penalty


def build_model(args):
    device = torch.device(args.device)
    model = Network(
        feats_size=args.feats_size,
        output_class=args.num_classes,
        dropout=args.dropout,
        big_lambda=args.big_lambda,
        selection_strategy=args.selection_strategy,
        task='classification',
        num_classes=args.num_classes,
        use_graph=args.use_graph,
        mamba_depth=args.mamba_depth,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        graph_model=args.graph_model,
        graph_hidden=args.graph_hidden,
        graph_out=args.graph_out,
        graph_dropout=args.graph_dropout,
        fuse_type=args.fuse_type,
        use_clinical=False,
    ).to(device)
    return model


def train_one_epoch(args, model,
                    optimizer,
                    criterion,
                    train_loader,
                    device,
                    current_epoch=0):
    model.train()
    epoch_loss = 0.0
    epoch_l2_penalty = 0.0
    all_predictions = []
    all_labels = []
    train_correct = 0
    train_total = 0
    

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        patch_features_list = batch['patch_features']
        labels_list = batch['label']  # Classification labels

        for j in range(len(patch_features_list)):
            bag_feats = patch_features_list[j].to(device)
            label = labels_list[j].long().view(1).to(device)
            graph_j = None
            if args.use_graph:
                if isinstance(batch.get('graph_data'), list):
                    graph_j = batch['graph_data'][j]
                else:
                    graph_j = batch.get('graph_data')
            out = model(bag_feats, graph_j, None)
            bag_pred = out['bag_prediction']  # [1, num_classes]
            patch_scores = out['patch_scores']  # [N, num_classes]

            # Track predictions and labels for metrics
            all_predictions.append(bag_pred.detach().cpu())
            all_labels.append(label.detach().cpu())
            
            probs = torch.softmax(bag_pred, dim=-1)
            predicted = torch.argmax(probs, dim=-1).item()
            
            true_label = int(label.item())
            train_correct += (predicted == true_label)
            train_total += 1

            # Optimize per bag
            optimizer.zero_grad()
            
            bag_loss = criterion(bag_pred, label)
            max_prediction, _ = torch.max(patch_scores, dim=0)  # [num_classes]
            max_loss = criterion(max_prediction.unsqueeze(0), label)
            main_loss = 0.5 * bag_loss + 0.5 * max_loss
            l2_penalty = 0.0
            if args.use_l2_reg:
                l2_penalty = compute_l2_penalty(model, args, current_epoch, args.epochs)
            total_loss = main_loss + l2_penalty
            total_loss.backward()
            optimizer.step()

            epoch_loss += (bag_loss.item() if isinstance(bag_loss, torch.Tensor) else 0.0)
            epoch_l2_penalty += l2_penalty.item() if isinstance(l2_penalty, torch.Tensor) else l2_penalty

        # Calculate running metrics for progress bar (like train.py)
        running_acc = train_correct / train_total if train_total > 0 else 0.0
        running_auc = 0.0
        
        # Calculate running AUC for binary classification
        if args.num_classes == 2 and len(all_predictions) > 0:
            try:
                running_preds = torch.cat(all_predictions, dim=0)
                running_labels = torch.stack(all_labels)
                running_probs = torch.softmax(running_preds, dim=1)[:, 1].cpu().numpy()
                running_labels_np = running_labels.cpu().numpy()
                # Check if both classes are present before calculating AUC
                if len(np.unique(running_labels_np)) > 1:
                    running_auc = roc_auc_score(running_labels_np, running_probs)
                else:
                    running_auc = 0.0
            except Exception as e:
                running_auc = 0.0
        
        if args.use_l2_reg:
            pbar.set_postfix({
                'loss': f"{epoch_loss/max(1, len(pbar)):.4f}", 
                'l2': f"{epoch_l2_penalty/max(1, len(pbar)):.4f}",
                'acc': f"{running_acc:.4f}",
                'auc': f"{running_auc:.4f}"
            })
        else:
            pbar.set_postfix({
                'loss': f"{epoch_loss/max(1, len(pbar)):.4f}",
                'acc': f"{running_acc:.4f}",
                'auc': f"{running_auc:.4f}"
            })

    # Calculate final accuracy and AUC (like train.py)
    accuracy = train_correct / train_total if train_total > 0 else 0.0
    
    # Calculate AUC (for binary classification)
    auc = 0.0
    if args.num_classes == 2 and len(all_predictions) > 0:
        try:
            predictions = torch.cat(all_predictions, dim=0)  # [total_samples, num_classes]
            labels = torch.stack(all_labels)  # [total_samples]
            # For binary classification, use the probability of the positive class
            probs = torch.softmax(predictions, dim=1)[:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy()
            # Check if both classes are present before calculating AUC
            if len(np.unique(labels_np)) > 1:
                auc = roc_auc_score(labels_np, probs)
            else:
                auc = 0.0
        except Exception as e:
            logging.warning(f"Failed to calculate AUC: {e}")
            auc = 0.0
    
    return epoch_loss / max(1, len(train_loader)), accuracy, epoch_l2_penalty / max(1, len(train_loader)), auc


def validate(args, model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Per-bag evaluation (no accumulation)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            patch_features_list = batch['patch_features']
            labels_list = batch['label']  # Classification labels

            for j in range(len(patch_features_list)):
                bag_feats = patch_features_list[j].to(device)
                label = labels_list[j].to(device)
                graph_j = None
                if args.use_graph:
                    if isinstance(batch.get('graph_data'), list):
                        graph_j = batch['graph_data'][j]
                    else:
                        graph_j = batch.get('graph_data')
                out = model(bag_feats, graph_j, None)
                bag_pred = out['bag_prediction']  # [1, num_classes] - classification logits
                ins_pred = out.get('patch_scores', out.get('instance_predictions', None))

                # Clinical features not used in classification
                all_predictions.append(bag_pred.detach().cpu())
                all_labels.append(label.detach().cpu())
                # Compute per-bag loss (0.5*bag + 0.5*max instance if available)
                bag_loss = criterion(bag_pred, label.long().unsqueeze(0))
                if ins_pred is not None and ins_pred.dim() == 2 and ins_pred.size(0) > 0:
                    max_prediction, _ = torch.max(ins_pred, dim=0)
                    max_loss = criterion(max_prediction.unsqueeze(0), label.long().unsqueeze(0))
                    main_loss = 0.5 * bag_loss + 0.5 * max_loss
                    val_loss += main_loss.item()
                else:
                    val_loss += bag_loss.item()

                # Per-bag accuracy (align with train.py style)
                if args.num_classes == 1:
                    pred_prob = torch.sigmoid(bag_pred).item()
                    predicted = 1 if pred_prob >= 0.5 else 0
                else:
                    probs = torch.softmax(bag_pred, dim=-1)
                    predicted = torch.argmax(probs, dim=-1).item()
                true_label = int(label.item())
                # Initialize counters if not yet defined in outer scope
                try:
                    val_correct
                except NameError:
                    val_correct = 0
                    val_total = 0
                val_correct += (predicted == true_label)
                val_total += 1
            
            # Calculate running accuracy and AUC for progress bar
            if len(all_predictions) > 0:
                running_acc = (val_correct / val_total) if val_total > 0 else 0.0
                running_preds = torch.cat(all_predictions, dim=0)
                running_labels = torch.stack(all_labels)
                # Calculate running AUC for binary classification
                running_auc = 0.0
                if args.num_classes == 2 and len(running_preds) > 1:
                    try:
                        running_probs = torch.softmax(running_preds, dim=1)[:, 1].cpu().numpy()
                        running_labels_np = running_labels.cpu().numpy()
                        # Check if both classes are present before calculating AUC
                        if len(np.unique(running_labels_np)) > 1:
                            running_auc = roc_auc_score(running_labels_np, running_probs)
                        else:
                            running_auc = 0.0
                    except Exception as e:
                        running_auc = 0.0
                
                pbar.set_postfix({
                    'loss': f"{val_loss/max(1, len(pbar)):.4f}",
                    'acc': f"{running_acc:.4f}",
                    'auc': f"{running_auc:.4f}"
                })
            else:
                pbar.set_postfix({'loss': f"{val_loss/max(1, len(pbar)):.4f}"})

        # No accumulation remainder handling

    # Final accuracy via counters
    try:
        final_acc = val_correct / val_total if val_total > 0 else 0.0
    except NameError:
        final_acc = 0.0

    # Calculate AUC (for binary classification)
    auc = 0.0
    if args.num_classes == 2 and len(all_predictions) > 0:
        try:
            predictions = torch.cat(all_predictions, dim=0)  # [total_samples, num_classes]
            labels = torch.stack(all_labels)  # [total_samples]
            # For binary classification, use the probability of the positive class
            probs = torch.softmax(predictions, dim=1)[:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy()
            # Check if both classes are present before calculating AUC
            if len(np.unique(labels_np)) > 1:
                auc = roc_auc_score(labels_np, probs)
            else:
                auc = 0.0
        except Exception as e:
            logging.warning(f"Failed to calculate AUC: {e}")
            auc = 0.0
    
    return val_loss / max(1, len(val_loader)), final_acc, auc


def _load_folds(args):
    import pandas as pd
    if os.path.isabs(args.fold_splits_csv):
        splits_path = args.fold_splits_csv
    else:
        splits_path = os.path.join(args.dataset_dir, args.fold_splits_csv)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Fold split file not found: {splits_path}")
    df = pd.read_csv(splits_path)
    logging.info(f"Load fold split file: {splits_path}")
    folds = {}
    id_col = args.id_column
    for fid, group in df.groupby('fold'):
        folds[int(fid)] = {
            'train': group.loc[group['split'] == 'train', id_col].astype(str).tolist(),
            'val': group.loc[group['split'] == 'val', id_col].astype(str).tolist(),
        }
    return folds


def _map_ids_to_indices(dataset: ClassificationDataset, fold_split: Dict[str, list]) -> Dict[str, list]:
    pid_to_indices = {}
    for idx in range(len(dataset)):
        pid = dataset[idx]['patient_id']
        pid_to_indices.setdefault(pid, []).append(idx)
    out = {'train': [], 'val': []}
    for split_name in ['train', 'val']:
        for pid in fold_split.get(split_name, []):
            if pid in pid_to_indices:
                out[split_name].extend(pid_to_indices[pid])
    return out


def parse_args():
    p = argparse.ArgumentParser(description='EfficientMIL Classification Training')
    # data
    p.add_argument('--dataset_dir', type=str, default='datasets/camelyon16')
    p.add_argument('--classification_file', type=str, default='label_c16.csv', help='CSV file with classification labels')
    p.add_argument('--classification_label_column', type=str, default='label', help='Column name for classification label')
    p.add_argument('--id_column', type=str, default='ID')
    p.add_argument('--fold_splits_csv', type=str, default='fold_splits.csv')
    p.add_argument('--id_mapping_file', type=str, default='id_mapping.csv')
    
    p.add_argument('--feats_size', type=int, default=1536)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--big_lambda', type=int, default=512)
    p.add_argument('--selection_strategy', type=str, default='aps', choices=['random-k','top-k','aps'])
    # task options
    p.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification task')
    # mamba
    p.add_argument('--mamba_depth', type=int, default=8)
    p.add_argument('--mamba_d_state', type=int, default=16)
    p.add_argument('--mamba_d_conv', type=int, default=4)
    p.add_argument('--mamba_expand', type=int, default=2)
    # train
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--lr_step_size', type=int, default=2)
    p.add_argument('--lr_gamma', type=float, default=0.6)
    p.add_argument('--min_lr', type=float, default=1e-8)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--num_folds', type=int, default=5)
    p.add_argument('--train_folds', type=str, default='all')
    p.add_argument('--save_dir', type=str, default='outputs/camelyon16')
    p.add_argument('--seed', type=int, default=42)
    # graph
    p.add_argument('--use_graph', action='store_true', help='Use WSI patch graphs if available')
    p.add_argument('--graph_model', type=str, default='gat', choices=['gcn','gat'], help='Graph encoder type')
    p.add_argument('--patch_features_dir', type=str, default='pt_files', help='Directory name for patch files')
    p.add_argument('--graph_features_dir', type=str, default='graph_files', help='Directory name for graph files')
    p.add_argument('--graph_hidden', type=int, default=256, help='GCN hidden channels')
    p.add_argument('--graph_out', type=int, default=256, help='GCN output channels for fusion')
    p.add_argument('--graph_dropout', type=float, default=0.1, help='GCN dropout')
    p.add_argument('--fuse_type', type=str, default='se', choices=['none', 'linear', 'se'], help='Feature fusion type after concatenation')
    # L2 regularization parameters
    p.add_argument('--use_l2_reg', action='store_true', default=True, help='Use L2 regularization')
    p.add_argument('--l2_lambda', type=float, default=1e-6, help='L2 regularization strength')
    p.add_argument('--l2_decay_schedule', type=str, default='decreasing', choices=['constant', 'increasing', 'decreasing'])
    p.add_argument('--max_l2_lambda', type=float, default=1e-4, help='Maximum L2 regularization strength')
    p.add_argument('--l2_penalty_type', type=str, default='classifier_only', choices=['all', 'classifier_only'])
    # resume
    p.add_argument('--resume', action='store_true', help='Resume training from last checkpoint if available')
    p.add_argument('--resume_path', type=str, default=None, help='Path to a last checkpoint to resume from (overrides default path)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    setup_logging(os.path.join(args.save_dir, 'log.txt'))
    set_random_seed(args.seed)
    
    # CUDA context management for multi-process training
    if torch.cuda.is_available():
        # Set unique CUDA context for this process
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        torch.cuda.set_device(device_id)
        
        # Clear CUDA cache to avoid conflicts
        torch.cuda.empty_cache()
        
        # Set process-specific CUDA context
        current_process = mp.current_process()
        logging.info(f"Process {current_process.name} (PID: {os.getpid()}) using CUDA device {args.device}")
        
        # Force CUDA context initialization
        dummy_tensor = torch.tensor([1.0]).cuda()
        del dummy_tensor
        torch.cuda.empty_cache()
    
    logging.info(f"Args: {args}")

    # dataset
    if args.use_graph:
        dataset = GraphClassificationDataset(
            dataset_dir=args.dataset_dir,
            patch_features_dir_name=args.patch_features_dir,
            graph_features_dir_name=args.graph_features_dir,
            classification_file=args.classification_file,
            id_column=args.id_column,
            classification_label_column=args.classification_label_column,
            use_graph_features=True,
            id_mapping_file=args.id_mapping_file,
        )
    else:
        dataset = ClassificationDataset(
            dataset_dir=args.dataset_dir,
            classification_file=args.classification_file,
            id_column=args.id_column,
            classification_label_column=args.classification_label_column,
            id_mapping_file=args.id_mapping_file,
        )
    logging.info(f"Classification dataset loaded: {len(dataset)} samples")

    folds = _load_folds(args)
    if args.train_folds is None or str(args.train_folds).lower() == 'all':
        folds_to_train = list(folds.keys())
    else:
        try:
            folds_to_train = [int(f.strip()) for f in str(args.train_folds).split(',') if f.strip()!='']
            for f in folds_to_train:
                if f not in folds:
                    raise ValueError(f"Invalid fold {f}. Available: {list(folds.keys())}")
        except Exception as e:
            logging.error(f"Invalid train_folds: {args.train_folds}. Error: {e}")
            return

    device = torch.device(args.device)
    fold_summaries = []
    for fid in folds_to_train:
        split = folds[fid]
        split_indices = _map_ids_to_indices(dataset, split)
        train_subset = Subset(dataset, split_indices['train'])
        val_subset = Subset(dataset, split_indices['val'])
        if args.use_graph:
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_with_graph, pin_memory=False, persistent_workers=False)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_with_graph, pin_memory=False, persistent_workers=False)
        else:
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False, persistent_workers=False)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False, persistent_workers=False)

        model = build_model(args)
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupStepLR(optimizer=optimizer, warmup_epochs=args.warmup_epochs, step_size=args.lr_step_size, gamma=args.lr_gamma, min_lr=args.min_lr)

        criterion = torch.nn.CrossEntropyLoss()

        best_accuracy = 0.0
        best_auc = 0.0
        best_epoch = -1
        patience_counter = 0
        fold_dir = os.path.join(args.save_dir, 'classification', f'fold_{fid}')
        os.makedirs(fold_dir, exist_ok=True)

        # Resume support
        start_epoch = 0
        if args.resume:
            last_ckpt = args.resume_path if args.resume_path else os.path.join(fold_dir, 'last.pth')
            if os.path.exists(last_ckpt):
                try:
                    ckpt = torch.load(last_ckpt, map_location='cpu')
                    model.load_state_dict(ckpt.get('model_state_dict', {}), strict=False)
                    if ckpt.get('optimizer_state_dict') is not None:
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    if ckpt.get('scheduler_state_dict') is not None:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    start_epoch = int(ckpt.get('epoch', -1)) + 1
                    best_accuracy = float(ckpt.get('best_accuracy', best_accuracy))
                    best_auc = float(ckpt.get('best_auc', best_auc))
                    patience_counter = int(ckpt.get('patience_counter', 0))
                    logging.info(f"Resumed from {last_ckpt}: start_epoch={start_epoch}, best_accuracy={best_accuracy:.4f}, best_auc={best_auc:.4f}, patience_counter={patience_counter}")
                except Exception as e:
                    logging.warning(f"Failed to resume from {last_ckpt}: {e}")

        else:
            initialize_weights(model)
            logging.info("Initialized model weights")

        for epoch in range(start_epoch, args.epochs):
            # Calculate current L2 lambda for logging
            current_l2_lambda = 0.0
            if args.use_l2_reg:
                if args.l2_decay_schedule == 'increasing':
                    progress = epoch / args.epochs
                    current_l2_lambda = min(args.l2_lambda + progress * (args.max_l2_lambda - args.l2_lambda), args.max_l2_lambda)
                elif args.l2_decay_schedule == 'decreasing':
                    progress = epoch / args.epochs
                    current_l2_lambda = max(args.l2_lambda - progress * (args.l2_lambda - args.max_l2_lambda * 0.1), args.max_l2_lambda * 0.1)
                else:
                    current_l2_lambda = args.l2_lambda
            
            l2_info = f" L2={current_l2_lambda:.6f}" if args.use_l2_reg else ""
            logging.info("-"*30 + f" Fold [{fid}/{args.num_folds}] Epoch [{epoch+1}/{args.epochs}] LR={optimizer.param_groups[0]['lr']:.6f}{l2_info} " + "-"*30)
            train_loss, train_acc, train_l2, train_auc = train_one_epoch(args, model, optimizer, criterion, train_loader, device, epoch)
            va_loss, val_acc, val_auc = validate(args, model, criterion, val_loader, device)
            if args.use_l2_reg:
                logging.info(f"Train Loss: {train_loss:.4f}, Train L2: {train_l2:.4f}, Train Accuracy: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
            else:
                logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
            logging.info(f"Val   Loss: {va_loss:.4f}, Val   Accuracy: {val_acc:.4f}, Val   AUC: {val_auc:.4f}")

            # save last
            last_path = os.path.join(fold_dir, 'last.pth')
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_acc,
                'val_auc': val_auc,
                'best_accuracy': best_accuracy,
                'best_auc': best_auc,
                'patience_counter': patience_counter,
            }
            if args.use_l2_reg:
                checkpoint_data['l2_penalty'] = train_l2
            torch.save(checkpoint_data, last_path)
            logging.info(f"Saved last to {last_path}")
            scheduler.step()

            if np.isnan(val_acc):
                patience_counter += 1
            elif val_acc > best_accuracy:
                best_accuracy = val_acc
                # Update best_auc when best_accuracy is updated
                if val_auc > best_auc:
                    best_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                best_path = os.path.join(fold_dir, 'best.pth')
                best_checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_accuracy': val_acc,
                    'val_auc': val_auc,
                    'best_accuracy': best_accuracy,
                    'best_auc': best_auc,
                    'patience_counter': patience_counter,
                }
                if args.use_l2_reg:
                    best_checkpoint_data['l2_penalty'] = train_l2
                torch.save(best_checkpoint_data, best_path)
                logging.info(f"Saved best to {best_path}")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logging.info("Early stopping")
                break

        # Record fold summary
        fold_summaries.append({'fold': fid, 'best_accuracy': float(best_accuracy), 'best_auc': float(best_auc), 'best_epoch': int(best_epoch)})

    # Log cross-fold summary
    if len(fold_summaries) > 0:
        logging.info("="*80)
        logging.info("CROSS-FOLD SUMMARY (Classification)")
        logging.info("="*80)
        for s in sorted(fold_summaries, key=lambda x: x['fold']):
            logging.info(f"Fold {s['fold']}: best_accuracy={s['best_accuracy']:.4f}, best_auc={s['best_auc']:.4f} at epoch {s['best_epoch']}")
        acc_list = [s['best_accuracy'] for s in fold_summaries if not np.isnan(s['best_accuracy'])]
        auc_list = [s['best_auc'] for s in fold_summaries if not np.isnan(s['best_auc'])]
        if len(acc_list) > 0:
            logging.info(f"Mean Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
        if len(auc_list) > 0:
            logging.info(f"Mean AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")


if __name__ == '__main__':
    main()
