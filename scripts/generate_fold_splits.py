#!/usr/bin/env python
import argparse
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn.model_selection import StratifiedKFold, KFold
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging


def generate_fold_splits(df: pd.DataFrame, num_folds: int, seed: int, save_path: str, events_override: np.ndarray | None = None, id_column: str = 'ID') -> pd.DataFrame:
    """
    Generate fold splits and save to a CSV file using stratified sampling.
    
    If events_override is provided, uses sklearn's StratifiedKFold with event/censoring
    status as the stratification variable to ensure consistent proportions across all folds.
    Otherwise, performs random K-Fold splitting (shuffle=True).
    """
    fold_data: list[dict] = []
    used_events: np.ndarray | None = None

    if events_override is not None:
        used_events = np.asarray(events_override, dtype=int)
        if used_events.ndim != 1 or len(used_events) != len(df):
            raise ValueError("events_override must be a 1-D array with length equal to dataset size")
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        iterator = skf.split(np.zeros(len(used_events)), used_events)
        desc_msg = "Generating fold splits (stratified by event status)"
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        iterator = kf.split(np.zeros(len(df)))
        desc_msg = "Generating fold splits (random K-Fold)"

    for fold_idx, (train_index, val_index) in tqdm(
        enumerate(iterator, start=1), 
        total=num_folds, 
        desc=desc_msg
    ):
        train_ids = df.iloc[train_index][id_column].astype(str).tolist()
        val_ids = df.iloc[val_index][id_column].astype(str).tolist()
        for pid in train_ids:
            fold_data.append({id_column: pid, 'fold': int(fold_idx), 'split': 'train'})
        for pid in val_ids:
            fold_data.append({id_column: pid, 'fold': int(fold_idx), 'split': 'val'})

    fold_df = pd.DataFrame(fold_data)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fold_df.to_csv(save_path, index=False)
    logging.info(f"Fold splits saved to {save_path}")

    # Print fold statistics; include event proportions only if stratified mode
    if used_events is not None:
        total_events = int(np.sum(used_events))
        total_censored = int(len(used_events) - total_events)
        logging.info(f"Overall dataset: {len(used_events)} samples, {total_events} events ({total_events/len(used_events)*100:.1f}%), {total_censored} censored ({total_censored/len(used_events)*100:.1f}%)")
        for fold in range(1, num_folds + 1):
            fold_subset = fold_df[fold_df['fold'] == fold]
            # Map back IDs in this fold to indices for event counts
            train_ids = fold_subset[fold_subset['split'] == 'train'][id_column].astype(str).tolist()
            val_ids = fold_subset[fold_subset['split'] == 'val'][id_column].astype(str).tolist()
            id_to_index = {str(v): i for i, v in enumerate(df[id_column].astype(str).tolist())}
            train_indices = np.array([id_to_index[i] for i in train_ids if i in id_to_index], dtype=int)
            val_indices = np.array([id_to_index[i] for i in val_ids if i in id_to_index], dtype=int)
            train_events = int(np.sum(used_events[train_indices]))
            val_events = int(np.sum(used_events[val_indices]))
            train_size = len(train_indices)
            val_size = len(val_indices)
            logging.info(f"Fold {fold}: train={train_size} (events={train_events}, {train_events/max(1,train_size)*100:.1f}%), val={val_size} (events={val_events}, {val_events/max(1,val_size)*100:.1f}%)")
    else:
        for fold in range(1, num_folds + 1):
            fold_subset = fold_df[fold_df['fold'] == fold]
            train_size = len(fold_subset[fold_subset['split'] == 'train'])
            val_size = len(fold_subset[fold_subset['split'] == 'val'])
            logging.info(f"Fold {fold}: train={train_size}, val={val_size}")

    return fold_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate fold_splits.csv using stratified K-Fold over event status')
    parser.add_argument('--dataset_dir', type=str, default='datasets/changhai_dataset', help='Root directory of the dataset')
    parser.add_argument('--patch_features_dir_name', type=str, default='pt_files', help='Patch features subdirectory name')
    parser.add_argument('--label_file', type=str, default='survival_data.csv', help='Survival labels CSV filename')
    parser.add_argument('--id_mapping_file', type=str, default='id_mapping.csv', help='ID mapping CSV filename')
    parser.add_argument('--id_column', type=str, default='ID', help='Column name containing the IDs in label CSV (will be used in output)')
    parser.add_argument('--time_col', type=str, default=None, help='Optional time column name in label CSV')
    parser.add_argument('--status_col', type=str, default=None, help='Optional status/event column name in label CSV (1=event,0=censored)')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds to generate (>=2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--output_csv', type=str, default='fold_splits.csv', help='Path to write fold_splits.csv')
    parser.add_argument('--log_file', type=str, default='./logs/generate_fold_splits.txt', help='Optional log file path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    if args.num_folds is None or args.num_folds < 2:
        raise ValueError("num_folds must be >= 2")

    # Load labels as DataFrame
    label_csv_path = os.path.join(args.dataset_dir, args.label_file)
    df = pd.read_csv(label_csv_path)
    logging.info(f"Labels loaded from {label_csv_path} with {len(df)} rows")

    # Determine events array for splitting if status column provided; otherwise use random K-Fold
    events_override: np.ndarray | None = None
    if args.status_col:
        if args.status_col not in df.columns:
            raise ValueError(f"status_col '{args.status_col}' not found in {label_csv_path}")
        events_override = df[args.status_col].astype(int).to_numpy()
        # time_col is not required for stratification; if provided, validate existence for user feedback
        if args.time_col:
            if args.time_col not in df.columns:
                raise ValueError(f"time_col '{args.time_col}' not found in {label_csv_path}")

    # Generate and save splits
    generate_fold_splits(df, 
                         args.num_folds, 
                         args.seed, 
                         os.path.join(args.dataset_dir, args.output_csv), 
                         events_override=events_override,
                         id_column=args.id_column)


if __name__ == '__main__':
    main()


