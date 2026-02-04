import argparse
import os
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any
import sys
sys.path.append("../")
from utils import setup_logging, set_random_seed
from data import SurvivalDataset, GraphSurvivalDataset, collate_fn, collate_fn_with_graph
from train_survival import build_model


def load_model_from_checkpoint(checkpoint_path: str, args, clinical_aux_dims=None):
    """Load model from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = build_model(args, clinical_aux_dims)
    
    state_dict = checkpoint['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    
    logging.info(f"Loaded model from {checkpoint_path}")
    return model


def infer(args, model, data_loader, device):
    """Run inference on the dataset."""
    model.eval()
    results = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Inferring')
        for batch in pbar:
            patch_features_list = batch['patch_features']
            patient_ids = batch.get('patient_id', [])
            time_list = batch.get('time', [])
            status_list = batch.get('status', [])
            
            for j in range(len(patch_features_list)):
                bag_feats = patch_features_list[j].to(device)
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
                    elif cv_list is not None and not isinstance(cv_list, list):
                        clinical_j = cv_list.to(device)
                
                out = model(bag_feats, graph_j, clinical_j)
                bag_pred = out['bag_prediction']
                risk_score = bag_pred.item() if bag_pred.numel() == 1 else bag_pred.cpu().numpy().flatten()[0]
                
                # Extract patient_id, time, and status
                # collate_fn returns these as lists
                patient_id = str(patient_ids[j])
                time_val = float(time_list[j].item()) if hasattr(time_list[j], 'item') else float(time_list[j])
                status_val = int(status_list[j].item()) if hasattr(status_list[j], 'item') else int(status_list[j])
                
                results.append({
                    'ID': patient_id,
                    'OS': time_val,
                    'Status': status_val,
                    'RiskScore': risk_score
                })
    
    return results


def parse_args():
    p = argparse.ArgumentParser(description='Survival Model Inference')
    # model checkpoint
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file')
    # data
    p.add_argument('--dataset_dir', type=str, default='datasets/tcga_luad')
    p.add_argument('--survival_file', type=str, default='survival_data_luad.csv', help='Path to survival data CSV file')
    p.add_argument('--id_column', type=str, default='ID')
    p.add_argument('--time_column', type=str, default='OS')
    p.add_argument('--event_column', type=str, default='Status')
    p.add_argument('--id_mapping_file', type=str, default='id_mapping.csv')
    # output
    p.add_argument('--output_file', type=str, default=None, help='Path to output CSV file')
    # model architecture (should match training config)
    p.add_argument('--feats_size', type=int, default=1536)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--big_lambda', type=int, default=512)
    p.add_argument('--selection_strategy', type=str, default='aps', choices=['random-k','top-k','aps'])
    # mamba
    p.add_argument('--mamba_depth', type=int, default=8)
    p.add_argument('--mamba_d_state', type=int, default=16)
    p.add_argument('--mamba_d_conv', type=int, default=4)
    p.add_argument('--mamba_expand', type=int, default=2)
    # device
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    # graph
    p.add_argument('--use_graph', action='store_true', help='Use WSI patch graphs if available')
    p.add_argument('--graph_model', type=str, default='gat', choices=['gcn','gat'], help='Graph encoder type')
    p.add_argument('--patch_features_dir', type=str, default='pt_files', help='Directory name for patch files')
    p.add_argument('--graph_features_dir', type=str, default='graph_files', help='Directory name for graph files')
    p.add_argument('--graph_hidden', type=int, default=512, help='GCN hidden channels')
    p.add_argument('--graph_out', type=int, default=256, help='GCN output channels for fusion')
    p.add_argument('--graph_dropout', type=float, default=0.1, help='GCN dropout')
    p.add_argument('--fuse_type', type=str, default='se', 
                   choices=['none','linear','se','cross_attention'], 
                   help='Feature fusion type after concatenation')
    # clinical options
    p.add_argument('--use_clinical', action='store_true', help='Use clinical features with clinical features encoder')
    p.add_argument('--clinical_hidden', type=int, default=256, help='Clinical feature vector dimension per clinical feature')
    p.add_argument('--clinical_norm', type=str, default='zscore', choices=['zscore', 'minmax'])
    p.add_argument('--clinical_num_cols', type=str, default='Age', help='Comma-separated numeric clinical columns from survival_data.csv')
    p.add_argument('--clinical_cat_cols', type=str, default='T,N,M,Gender', help='Comma-separated categorical clinical columns from survival_data.csv')
    # misc
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    if args.output_file is None:
        output_dir = os.path.dirname(args.checkpoint)
        args.output_file = os.path.join(output_dir, 'results.csv')
    else:
        output_dir = os.path.dirname(args.output_file)
        
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'infer_log.txt') if output_dir else 'infer_log.txt'
    setup_logging(log_file)
    set_random_seed(args.seed)
    
    logging.info(f"Args: {args}")
    
    # Determine if survival_file is absolute or relative
    if os.path.isabs(args.survival_file):
        survival_file_path = args.survival_file
    else:
        survival_file_path = os.path.join(args.dataset_dir, args.survival_file)
    
    if not os.path.exists(survival_file_path):
        raise FileNotFoundError(f"Survival file not found: {survival_file_path}")
    
    # Prepare clinical feature columns
    clinical_num_cols = [c.strip() for c in args.clinical_num_cols.split(',')] if args.clinical_num_cols else []
    clinical_cat_cols = [c.strip() for c in args.clinical_cat_cols.split(',')] if args.clinical_cat_cols else []
    
    # Load dataset
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
    
    logging.info(f"Loaded dataset: {len(dataset)} samples")
    
    # Get clinical_aux_dims from dataset if using clinical features
    clinical_aux_dims = None
    if args.use_clinical:
        clinical_aux_dims = getattr(dataset, 'clinical_aux_dims', [])
    
    # Create data loader
    if args.use_graph:
        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            collate_fn=collate_fn_with_graph
        )
    else:
        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            collate_fn=collate_fn
        )
    
    # Load model
    device = torch.device(args.device)
    model = load_model_from_checkpoint(args.checkpoint, args, clinical_aux_dims)
    model = model.to(device)
    
    # Run inference
    logging.info("Starting inference...")
    results = infer(args, model, data_loader, device)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output_file, index=False)
    logging.info(f"Saved inference results to {args.output_file}")
    logging.info(f"Total samples: {len(results)}")


if __name__ == '__main__':
    main()

