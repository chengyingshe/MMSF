import argparse
import os
import logging
from typing import Optional, Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import multiprocessing as mp


from utils import setup_logging, set_random_seed, WarmupStepLR
from data import SurvivalDataset, GraphSurvivalDataset, collate_fn, collate_fn_with_graph
from survival_utils import CoxLoss, compute_c_index
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

    return current_l2_lambda * l2_penalty


def build_model(args, clinical_aux_dims=None):
    device = torch.device(args.device)
    model = Network(
        feats_size=args.feats_size,
        output_class=1,
        dropout=args.dropout,
        big_lambda=args.big_lambda,
        selection_strategy=args.selection_strategy,
        task='survival',
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
        use_clinical=args.use_clinical,
        clinical_hidden=args.clinical_hidden,
        clinical_aux_dims=clinical_aux_dims,
    ).to(device)
    if args.pretrained_weight is not None:
        model.load_state_dict(torch.load(args.pretrained_weight), strict=False)
        logging.info(f"Loaded pretrained weight from {args.pretrained_weight}")
    else:
        initialize_weights(model)
        logging.info("Initialized weights")
    return model


def train_one_epoch(args, model,
                    optimizer,
                    cox_loss_fn,
                    train_loader,
                    device,
                    current_epoch=0,
                    clinical_embedding_loss_fn=None):
    model.train()
    epoch_loss = 0.0
    epoch_l2_penalty = 0.0
    all_risks = []
    all_times = []
    all_events = []
    all_clinical_losses = []
    
    # Accumulate samples for Cox loss computation
    accumulated_risks = []
    accumulated_times = []
    accumulated_events = []
    accumulation_size = 0
    target_accumulation = max(8, args.batch_size * 4)  # Accumulate at least 8 samples for meaningful Cox loss

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        patch_features_list = batch['patch_features']
        time_list = batch['time']
        status_list = batch['status']
        
        loss_components = {}
        for j in range(len(patch_features_list)):
            bag_feats = patch_features_list[j].to(device)
            t = time_list[j].to(device)
            e = status_list[j].to(device)
            graph_j = None
            if args.use_graph:
                if isinstance(batch.get('graph_data'), list):
                    graph_j = batch['graph_data'][j]
                else:
                    graph_j = batch.get('graph_data')
            clinical_j = None
            if args.use_clinical:
                cv_list = batch.get('clinical_vector')
                if isinstance(cv_list, list) and len(cv_list) > j:
                    clinical_j = cv_list[j].to(device)
            out = model(bag_feats, graph_j, clinical_j)
            bag_pred = out['bag_prediction']
            # bag_pred: [1,1] -> risk score
            accumulated_risks.append(bag_pred.view(-1))
            accumulated_times.append(t.view(-1))
            accumulated_events.append(e.view(-1))
            accumulation_size += 1
            
            all_risks.append(bag_pred.detach().cpu().view(-1))
            all_times.append(t.detach().cpu().view(-1))
            all_events.append(e.detach().cpu().view(-1))
            # Track clinical reconstruction loss via provided loss_fn
            if args.use_clinical and clinical_j is not None and 'clinical_recon' in out and callable(clinical_embedding_loss_fn):
                pred = out['clinical_recon']  # [1, sum_aux]
                target = clinical_j.view(1, -1)
                clinical_loss_sample = clinical_embedding_loss_fn(pred, target)
                all_clinical_losses.append(clinical_loss_sample.detach().cpu())

        # Compute loss when we have enough samples
        if accumulation_size >= target_accumulation:
            optimizer.zero_grad()
            risk_vec = torch.cat(accumulated_risks)
            t_vec = torch.cat(accumulated_times)
            e_vec = torch.cat(accumulated_events)
            
            # Compute Cox loss
            cox_loss = cox_loss_fn(risk_vec, t_vec, e_vec)
            loss_components['cox_loss'] = cox_loss.item()

            clinical_loss_val = 0.0
            if args.use_clinical and len(all_clinical_losses) > 0:
                clinical_loss_val = torch.mean(torch.stack(all_clinical_losses)).to(device) * args.clinical_loss_weight
                loss_components['clinical_loss'] = clinical_loss_val.item()
            
            # Compute L2 regularization penalty if enabled
            l2_penalty = 0.0
            if args.use_l2_reg:
                l2_penalty = compute_l2_penalty(model, args, current_epoch, args.epochs)
                loss_components['l2_loss'] = l2_penalty.item()
            
            total_loss = cox_loss + (clinical_loss_val if isinstance(clinical_loss_val, torch.Tensor) else 0.0) + l2_penalty
            loss_components['total_loss'] = total_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += cox_loss.item()
            epoch_l2_penalty += l2_penalty.item() if isinstance(l2_penalty, torch.Tensor) else l2_penalty
            
            # Reset accumulation
            accumulated_risks = []
            accumulated_times = []
            accumulated_events = []
            accumulation_size = 0
        
            pbar.set_postfix(loss_components)

    # Handle remaining samples
    if accumulation_size > 0:
        optimizer.zero_grad()
        risk_vec = torch.cat(accumulated_risks)
        t_vec = torch.cat(accumulated_times)
        e_vec = torch.cat(accumulated_events)
        
        # Compute Cox loss
        cox_loss = cox_loss_fn(risk_vec, t_vec, e_vec)
        
        # Compute L2 regularization penalty if enabled
        l2_penalty = 0.0
        if args.use_l2_reg:
            l2_penalty = compute_l2_penalty(model, args, current_epoch, args.epochs)
        
        # Total loss = Cox loss + reconstruction loss + L2 penalty
        clinical_loss_val = 0.0
        if args.use_clinical and len(all_clinical_losses) > 0:
            clinical_loss_val = torch.mean(torch.stack(all_clinical_losses)).to(device) * args.clinical_loss_weight
        total_loss = cox_loss + clinical_loss_val + l2_penalty
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += cox_loss.item()
        epoch_l2_penalty += l2_penalty.item() if isinstance(l2_penalty, torch.Tensor) else l2_penalty

    risks = torch.cat(all_risks)
    times = torch.cat(all_times)
    events = torch.cat(all_events)
    c_train = compute_c_index(risks, times, events)
    
    return epoch_loss / max(1, len(train_loader)), c_train, epoch_l2_penalty / max(1, len(train_loader))


def validate(args, model, cox_loss_fn, val_loader, device, clinical_embedding_loss_fn=None):
    model.eval()
    val_loss = 0.0
    all_risks = []
    all_times = []
    all_events = []
    all_clinical_losses = []
    
    # Accumulate samples for Cox loss computation
    accumulated_risks = []
    accumulated_times = []
    accumulated_events = []
    accumulation_size = 0
    target_accumulation = max(8, args.batch_size * 4)  # Accumulate at least 8 samples for meaningful Cox loss
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            patch_features_list = batch['patch_features']
            time_list = batch['time']
            status_list = batch['status']
            loss_components = {}
            for j in range(len(patch_features_list)):
                bag_feats = patch_features_list[j].to(device)
                t = time_list[j].to(device)
                e = status_list[j].to(device)
                graph_j = None
                if args.use_graph:
                    if isinstance(batch.get('graph_data'), list):
                        graph_j = batch['graph_data'][j]
                    else:
                        graph_j = batch.get('graph_data')
                clinical_j = None
                if args.use_clinical:
                    cv_list = batch.get('clinical_vector')
                    if isinstance(cv_list, list) and len(cv_list) > j:
                        clinical_j = cv_list[j].to(device)
                out = model(bag_feats, graph_j, clinical_j)
                bag_pred = out['bag_prediction']
                if args.use_clinical and 'clinical_recon' in out and callable(clinical_embedding_loss_fn) and clinical_j is not None:
                    pred = out['clinical_recon']
                    target = clinical_j.view(1, -1)
                    all_clinical_losses.append(clinical_embedding_loss_fn(pred, target).detach().cpu())
                    loss_components['clinical_loss'] = clinical_embedding_loss_fn(pred, target).item()
                accumulated_risks.append(bag_pred.view(-1))
                accumulated_times.append(t.view(-1))
                accumulated_events.append(e.view(-1))
                accumulation_size += 1
                
                all_risks.append(bag_pred.detach().cpu().view(-1))
                all_times.append(t.detach().cpu().view(-1))
                all_events.append(e.detach().cpu().view(-1))

            # Compute loss when we have enough samples
            if accumulation_size >= target_accumulation:
                risk_vec = torch.cat(accumulated_risks)
                t_vec = torch.cat(accumulated_times)
                e_vec = torch.cat(accumulated_events)
                loss = cox_loss_fn(risk_vec, t_vec, e_vec)
                loss_components['cox_loss'] = loss.item()
                val_loss += loss.item()
                
                # Reset accumulation
                accumulated_risks = []
                accumulated_times = []
                accumulated_events = []
                accumulation_size = 0
                
            pbar.set_postfix(loss_components)

        # Handle remaining samples
        if accumulation_size > 0:
            risk_vec = torch.cat(accumulated_risks)
            t_vec = torch.cat(accumulated_times)
            e_vec = torch.cat(accumulated_events)
            loss = cox_loss_fn(risk_vec, t_vec, e_vec)
            val_loss += loss.item()

    risks = torch.cat(all_risks)
    times = torch.cat(all_times)
    events = torch.cat(all_events)
    c_val = compute_c_index(risks, times, events)
    return val_loss / max(1, len(val_loader)), c_val


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


def _map_ids_to_indices(dataset: SurvivalDataset, fold_split: Dict[str, list]) -> Dict[str, list]:
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
    p = argparse.ArgumentParser(description='EfficientMIL Survival Training')
    # data
    p.add_argument('--dataset_dir', type=str, default='datasets/tcga_luad')
    p.add_argument('--survival_file', type=str, default='survival_data_luad.csv')
    p.add_argument('--id_column', type=str, default='ID')
    p.add_argument('--time_column', type=str, default='OS')
    p.add_argument('--event_column', type=str, default='Status')
    p.add_argument('--fold_splits_csv', type=str, default='fold_splits.csv')
    p.add_argument('--id_mapping_file', type=str, default='id_mapping.csv')
    
    p.add_argument('--feats_size', type=int, default=1536)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--big_lambda', type=int, default=512)
    p.add_argument('--selection_strategy', type=str, default='aps', choices=['random-k','top-k','aps'])
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
    p.add_argument('--pretrained_weight', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='outputs/tcga_luad-graph_mamba')
    p.add_argument('--seed', type=int, default=42)
    # graph
    p.add_argument('--use_graph', action='store_true', help='Use WSI patch graphs if available')
    p.add_argument('--graph_model', type=str, default='gat', choices=['gcn','gat'], help='Graph encoder type')
    p.add_argument('--patch_features_dir', type=str, default='pt_files', help='Directory name for patch files')
    p.add_argument('--graph_features_dir', type=str, default='graph_files', help='Directory name for graph files')
    p.add_argument('--graph_hidden', type=int, default=256, help='GCN hidden channels')
    p.add_argument('--graph_out', type=int, default=256, help='GCN output channels for fusion')
    p.add_argument('--graph_dropout', type=float, default=0.1, help='GCN dropout')
    p.add_argument('--fuse_type', type=str, default='se', 
                   choices=['none','linear','se','cross_attention'], 
                   help='Feature fusion type after concatenation')
    # clinical options
    p.add_argument('--use_clinical', action='store_true', help='Use clinical features with clinical features encoder')
    p.add_argument('--clinical_hidden', type=int, default=512, help='Clinical feature vector dimension per clinical feature')
    p.add_argument('--clinical_loss_weight', type=float, default=0.1, help='Weight for clinical encoder reconstruction loss')
    p.add_argument('--clinical_norm', type=str, default='zscore', choices=['zscore', 'minmax'])
    p.add_argument('--clinical_num_cols', type=str, default='Age', help='Comma-separated numeric clinical columns from survival_data.csv')
    p.add_argument('--clinical_cat_cols', type=str, default='T,N,M,Gender', help='Comma-separated categorical clinical columns from survival_data.csv')
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
    clinical_num_cols = [c.strip() for c in args.clinical_num_cols.split(',')]
    clinical_cat_cols = [c.strip() for c in args.clinical_cat_cols.split(',')]

    if args.use_graph:
        dataset = GraphSurvivalDataset(
            dataset_dir=args.dataset_dir,
            patch_features_dir_name=args.patch_features_dir,
            graph_features_dir_name=args.graph_features_dir,
            survival_file=args.survival_file,
            id_column=args.id_column,
            time_column=args.time_column,
            event_column=args.event_column,
            use_graph_features=True,
            id_mapping_file=args.id_mapping_file,
            clinical_cat_cols=clinical_cat_cols,
            clinical_num_cols=clinical_num_cols,
            clinical_norm=args.clinical_norm,
        )
    else:
        dataset = SurvivalDataset(
            dataset_dir=args.dataset_dir,
            survival_file=args.survival_file,
            id_column=args.id_column,
            time_column=args.time_column,
            event_column=args.event_column,
            id_mapping_file=args.id_mapping_file,
            clinical_cat_cols=clinical_cat_cols,
            clinical_num_cols=clinical_num_cols,
            clinical_norm=args.clinical_norm,
        )
    logging.info(f"Survival dataset loaded: {len(dataset)} samples")

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
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_with_graph)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_with_graph)
        else:
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        
        if args.use_clinical:
            base_ds = train_subset.dataset
            clinical_embedding_loss_fn = build_clinical_embedding_loss_fn(base_ds)
            clinical_aux_dims = getattr(base_ds, 'clinical_aux_dims', [])
        else:
            clinical_embedding_loss_fn = None
            clinical_aux_dims = None
        model = build_model(args, clinical_aux_dims)
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupStepLR(optimizer=optimizer, warmup_epochs=args.warmup_epochs, step_size=args.lr_step_size, gamma=args.lr_gamma, min_lr=args.min_lr)
        cox_loss_fn = CoxLoss()

        best_c = 0.0
        best_epoch = -1
        patience_counter = 0
        fold_dir = os.path.join(args.save_dir, 'survival', f'fold_{fid}')
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
                    best_c = float(ckpt.get('best_c_index', best_c))
                    patience_counter = int(ckpt.get('patience_counter', 0))
                    logging.info(f"Resumed from {last_ckpt}: start_epoch={start_epoch}, best_c_index={best_c:.4f}, patience_counter={patience_counter}")
                except Exception as e:
                    logging.warning(f"Failed to resume from {last_ckpt}: {e}")

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
            tr_loss, tr_c, tr_l2 = train_one_epoch(args, model, optimizer, cox_loss_fn, train_loader, device, epoch, clinical_embedding_loss_fn)
            va_loss, va_c = validate(args, model, cox_loss_fn, val_loader, device, clinical_embedding_loss_fn)
            if args.use_l2_reg:
                logging.info(f"Train Cox Loss: {tr_loss:.4f}, Train L2: {tr_l2:.4f}, Train C-index: {tr_c:.4f}")
            else:
                logging.info(f"Train Loss: {tr_loss:.4f}, Train C-index: {tr_c:.4f}")
            logging.info(f"Val   Loss: {va_loss:.4f}, Val   C-index: {va_c:.4f}")

            # save last
            last_path = os.path.join(fold_dir, 'last.pth')
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_c_index': va_c,
                'best_c_index': best_c,
                'patience_counter': patience_counter,
            }
            if args.use_l2_reg:
                checkpoint_data['l2_penalty'] = tr_l2
            torch.save(checkpoint_data, last_path)
            logging.info(f"Saved last to {last_path}")
            scheduler.step()

            if np.isnan(va_c):
                patience_counter += 1
            elif va_c >= best_c:
                best_c = va_c
                best_epoch = epoch
                patience_counter = 0
                best_path = os.path.join(fold_dir, 'best.pth')
                best_checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_c_index': va_c,
                    'best_c_index': best_c,
                    'patience_counter': patience_counter,
                }
                if args.use_l2_reg:
                    best_checkpoint_data['l2_penalty'] = tr_l2
                torch.save(best_checkpoint_data, best_path)
                logging.info(f"Saved best to {best_path}")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logging.info("Early stopping")
                break

        # Record fold summary
        fold_summaries.append({'fold': fid, 'best_c_index': float(best_c), 'best_epoch': int(best_epoch)})

    # Log cross-fold summary
    if len(fold_summaries) > 0:
        logging.info("="*80)
        logging.info("CROSS-FOLD SUMMARY (Survival)")
        logging.info("="*80)
        for s in sorted(fold_summaries, key=lambda x: x['fold']):
            logging.info(f"Fold {s['fold']}: best_c_index={s['best_c_index']:.4f} at epoch {s['best_epoch']}")
        c_list = [s['best_c_index'] for s in fold_summaries if not np.isnan(s['best_c_index'])]
        if len(c_list) > 0:
            logging.info(f"Mean C-index: {np.mean(c_list):.4f} ± {np.std(c_list):.4f}")


if __name__ == '__main__':
    main()


