#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import re
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging

def create_id_mapping(dataset_dir, csv_file, pt_dir, id_column='ID', output_file='id_mapping.csv'):
    """Create mapping between CSV IDs and actual pt file names"""
    
    csv_file = os.path.join(dataset_dir, csv_file)
    pt_dir = os.path.join(dataset_dir, pt_dir)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    csv_ids = df[id_column].astype(str).tolist()
    
    # Get all pt files
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    
    logging.info(f"Found {len(csv_ids)} IDs in CSV")
    logging.info(f"Found {len(pt_files)} pt files")
    
    # Create mapping
    mapping = {}
    missing_files = []
    
    for csv_id in csv_ids:
        # Try to find matching pt file
        found = False
        for pt_file in pt_files:
            if csv_id in pt_file:
                mapping[csv_id] = pt_file
                found = True
                break
        
        if not found:
            missing_files.append(csv_id)
    
    logging.info(f"Mapping created:")
    logging.info(f"  Matched files: {len(mapping)}")
    logging.info(f"  Missing files: {len(missing_files)}")
    
    if missing_files:
        logging.info(f"  Missing {len(missing_files)} IDs: {missing_files[:5]}...")
    
    # Save mapping
    mapping_df = pd.DataFrame([
        {'csv_id': k, 'pt_filename': v} for k, v in mapping.items()
    ])
    output_path = os.path.join(dataset_dir, output_file)
    mapping_df.to_csv(output_path, index=False)
    
    logging.info(f"Mapping saved to {output_path}")
    
    return mapping

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create mapping between CSV IDs and actual pt file names')
    parser.add_argument('--dataset_dir', type=str, default='datasets/tcga_paad', help='Root directory of the dataset')
    parser.add_argument('--csv_file', type=str, default='survival_data_paad.csv', help='CSV file containing IDs')
    parser.add_argument('--pt_dir', type=str, default='pt_files', help='Directory containing pt files')
    parser.add_argument('--id_column', type=str, default='ID', help='Column name containing the IDs in CSV file')
    parser.add_argument('--output_file', type=str, default='id_mapping.csv', help='Output mapping CSV filename')
    parser.add_argument('--log_file', type=str, default='./logs/create_id_mapping.txt', help='Optional log file path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    
    # Validate that dataset directory exists
    if not os.path.exists(args.dataset_dir):
        raise ValueError(f"Dataset directory '{args.dataset_dir}' does not exist")
    
    # Validate that CSV file exists
    csv_path = os.path.join(args.dataset_dir, args.csv_file)
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file '{csv_path}' does not exist")
    
    # Validate that pt directory exists
    pt_path = os.path.join(args.dataset_dir, args.pt_dir)
    if not os.path.exists(pt_path):
        raise ValueError(f"PT directory '{pt_path}' does not exist")
    
    # Load CSV and validate ID column
    df = pd.read_csv(csv_path)
    if args.id_column not in df.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in CSV file")
    
    logging.info(f"Creating ID mapping for dataset: {args.dataset_dir}")
    logging.info(f"CSV file: {args.csv_file}")
    logging.info(f"PT directory: {args.pt_dir}")
    logging.info(f"ID column: {args.id_column}")
    logging.info(f"Output file: {args.output_file}")
    
    # Create the mapping
    create_id_mapping(args.dataset_dir, args.csv_file, args.pt_dir, args.id_column, args.output_file)


if __name__ == "__main__":
    main() 