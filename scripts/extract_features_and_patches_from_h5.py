#!/usr/bin/env python3
"""
H5 to PT Converter Script
将H5文件中的features提取并转换为PyTorch .pt文件

使用方法:
python convert_h5_to_pt.py --input_dir /path/to/h5/files --pt_files_output_dir /path/to/output
"""

from glob import glob
import os
import h5py
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging(log_file='convert_h5_to_pt.log', verbose=False):
    """设置日志配置"""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.CRITICAL,  # 只显示严重错误
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    return logging.getLogger(__name__)

def extract_sample_name(h5_filename):
    """
    从H5文件名中提取样本名称
    例子: TCGA-2J-AAB4-01Z-00-DX1.480BBC89-87E5-4F6B-B0E9-A6A8A7F4DB5E.h5
    提取: TCGA-2J-AAB4-01Z-00-DX1
    """
    # 移除.h5扩展名
    name_without_ext = h5_filename.replace('.h5', '')
    
    # 按点分割，取第一部分（TCGA样本ID部分）
    parts = name_without_ext.split('.')
    sample_name = parts[0]
    
    return sample_name

def extract_features_patches_from_h5(input_file_path, pt_files_output_path, patches_output_path, extract_features, extract_patches, verbose):
    try:
        with h5py.File(input_file_path, 'r') as h5_file:
            # 检查是否包含features键
            if extract_features:
                if os.path.exists(pt_files_output_path):
                    if verbose:
                        logging.info(f"文件已存在: {pt_files_output_path}")
                
                if 'features' not in h5_file.keys():
                    if verbose:
                        logging.warning(f"文件 {input_file_path} 中没有找到 'features' 键")
                        logging.info(f"可用键: {list(h5_file.keys())}")
                
                # 提取features数据
                features = h5_file['features'][:]
                features_tensor = torch.from_numpy(features)
                torch.save(features_tensor, pt_files_output_path)             
            
            if extract_patches:
                if os.path.exists(patches_output_path):
                    if verbose:
                        logging.info(f"文件已存在: {patches_output_path}")
                
                if 'coords_patching' in h5_file.keys():
                    coords_patching = h5_file['coords_patching'][:]
                    with h5py.File(patches_output_path, 'w') as f:
                        f.create_dataset('coords', data=coords_patching)
                elif 'coords' in h5_file.keys():
                    coords = h5_file['coords'][:]
                    if len(coords.shape) == 3:
                        coords = coords[0]
                    with h5py.File(patches_output_path, 'w') as f:
                        f.create_dataset('coords', data=coords)
                
                if 'coords_patching' not in h5_file.keys() and 'coords' not in h5_file.keys():
                    if verbose:
                        logging.warning(f"文件 {input_file_path} 中没有找到 'coords_patching' 和 'coords' 键")
                        logging.info(f"可用键: {list(h5_file.keys())}")

    except Exception as e:
        logging.error(f"转换失败 {input_file_path}: {str(e)}")

def batch_extract_features_patches_from_h5(input_dir, pt_files_dir, patches_dir, extract_features, extract_patches, pattern="*.h5", verbose=False):
    os.makedirs(pt_files_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    h5_files = glob(os.path.join(input_dir, pattern))
    
    if not h5_files:
        if verbose:
            logging.warning(f"在目录 {input_dir} 中没有找到匹配 {pattern} 的文件")
        return
    
    if verbose:
        logging.info(f"找到 {len(h5_files)} 个H5文件待转换")
    
    # 统计信息
    success_count = 0
    failed_count = 0
    
    # 批量转换
    for h5_file in tqdm(h5_files, desc="Extracting"):
        h5_filename = os.path.basename(h5_file)
        
        file_id = h5_filename.split('.')[0]
        pt_files_path = os.path.join(pt_files_dir, file_id + '.pt')
        patches_path = os.path.join(patches_dir, file_id + '.h5')
        
        extract_features_patches_from_h5(str(h5_file), pt_files_path, patches_path, extract_features, extract_patches, verbose)
        
def main():
    parser = argparse.ArgumentParser(description='将H5文件中的features提取并转换为PyTorch .pt文件')
    parser.add_argument('--input_dir', default='/home/scy/changhai_project/wsi/datasets/wsi_features/UNI2-h_features/TCGA/TCGA-PAAD',
                        help='输入H5文件目录路径')
    parser.add_argument('--pt_files', default='./converted_pt_files/', help='输出pt_files目录路径')
    parser.add_argument('--patches', default='./converted_patches_files/', help='输出patches目录路径')
    parser.add_argument('--extract_features', action='store_true', help='是否提取features')
    parser.add_argument('--extract_patches', action='store_true', help='是否提取patches')
    parser.add_argument('--pattern', default='*.h5', help='文件匹配模式 (默认: *.h5)')
    parser.add_argument('--log_file', default='convert_h5_to_pt.log', help='日志文件路径')
    parser.add_argument('--verbose', 
                        action='store_true',
                        help='输出详细的日志信息')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_file, args.verbose)
    
    if args.verbose:
        logger.info("开始H5到PT转换任务")
        logger.info(f"输入目录: {args.input_dir}")
        logger.info(f"输出pt_files目录: {args.pt_files}")
        logger.info(f"输出patches目录: {args.patches}")
        logger.info(f"提取features: {args.extract_features}")
        logger.info(f"提取patches: {args.extract_patches}")

    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return

    batch_extract_features_patches_from_h5(args.input_dir, args.pt_files, args.patches, args.extract_features, args.extract_patches, args.pattern, args.verbose)
    

if __name__ == "__main__":
    main() 