import h5py
import csv
import torch
import torch.nn as nn
from torch_geometric.data import Data as geomData
from torch.utils.data import Dataset, ConcatDataset 
from torch.utils.data.dataset import random_split
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.utils import shuffle
import sys, argparse,os, datetime
from tqdm import tqdm
from torch.autograd import profiler
import time
import pickle
import itertools
import skimage as ski
from skimage import graph
import joblib
import matplotlib.pyplot as plt
import json
import sys, argparse, os, copy, itertools, glob, datetime
from torch_geometric.utils import add_self_loops

import math
import PIL
from PIL import Image

try:
    from openslide import OpenSlide, OpenSlideError
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: openslide not available. Some visualization functions may not work.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PatchGraphs(Dataset):
    """构建基于相邻补丁的图结构"""
    def __init__(self, dataset_dir, args) -> None:
        super(PatchGraphs, self).__init__()
        self.dataset_dir = dataset_dir
        self.patches_dir = os.path.join(dataset_dir, 'patches')
        self.pt_files_dir = os.path.join(dataset_dir, 'pt_files')
        self.graph_save_dir = os.path.join(dataset_dir, 'graph_files')
        self.args = args
        
        # 创建图保存目录
        os.makedirs(self.graph_save_dir, exist_ok=True)
        
        # 获取所有H5文件列表
        self.h5_files = [f for f in os.listdir(self.patches_dir) 
                        if f.endswith(".h5") and os.path.isfile(os.path.join(self.patches_dir, f))]
        self.h5_files.sort()  # 确保顺序一致
        
        print(f"Found {len(self.h5_files)} H5 files to process")

    def pt2graph_adjacentpatch(self, coords, features, radius=1):
        """将补丁坐标和特征转换为相邻补丁图"""
        edge_list = []
        max_neighbors = 0
        min_neighbors = 100
        count_isolated = 0
        count_edges = 0

        # 转换为tensor
        coords = torch.tensor(coords, device=device, dtype=torch.float32)
        features = torch.tensor(features, device=device, dtype=torch.float32)

        for i, current_patch in enumerate(coords):
            # 计算相邻补丁（基于patch_size的距离）
            adjacent_mask = torch.all(torch.abs(coords - current_patch) <= (self.args.patch_size * radius), dim=1)

            if max_neighbors < torch.sum(adjacent_mask):
                max_neighbors = torch.sum(adjacent_mask) 

            if min_neighbors >= torch.sum(adjacent_mask):
                min_neighbors = torch.sum(adjacent_mask) 
                if min_neighbors == 1:
                    count_isolated += 1

            # 排除自己
            adjacent_mask[i] = False
            num_neighbors = torch.sum(adjacent_mask)
            count_edges += torch.sum(adjacent_mask)

            # 获取邻居索引
            neighbor_indices = torch.nonzero(adjacent_mask).squeeze()
            if neighbor_indices.numel() == 0:
                continue
            elif neighbor_indices.numel() == 1:
                edge_list.append([i, neighbor_indices.item()])
            else:
                edge_list.extend([[i, idx.item()] for idx in neighbor_indices])

        print(f"Max neighbors: {max_neighbors}, Min neighbors: {min_neighbors}, Isolated patches: {count_isolated}")

        # 构建边索引
        if len(edge_list) == 0:
            # 如果没有边，创建自环
            edge_index = torch.tensor([[i for i in range(len(coords))], 
                                     [i for i in range(len(coords))]], 
                                    dtype=torch.long, device=device)
        else:
            edge_index = torch.LongTensor(np.array(edge_list).T).to(device)
            # 添加自环
            edge_index, _ = add_self_loops(edge_index, num_nodes=coords.size(0))

        # 创建PyG图数据
        G = geomData(x=features,
                     edge_index=edge_index,
                     centroid=coords)
        
        return G

    def create_patch_graphs(self):
        """为所有文件创建补丁图"""
        print('\n=== Creating patch graphs ===\n')
        
        for i, h5_fname in enumerate(self.h5_files):
            print(f'Processing {i+1}/{len(self.h5_files)}: {h5_fname}')
            # 构建文件路径
            file_id = h5_fname.split('.')[0]
            h5_path = os.path.join(self.patches_dir, h5_fname)
            pt_path = os.path.join(self.pt_files_dir, file_id + '.pt')
            graph_fname = file_id + '.pt'
            graph_path = os.path.join(self.graph_save_dir, graph_fname)
            
            # 检查图文件是否已存在
            if os.path.exists(graph_path):
                print(f'Graph already exists: {graph_fname}')
                continue
                
            # 检查必要文件是否存在
            if not os.path.exists(h5_path):
                print(f'H5 file not found: {h5_path}')
                continue
            if not os.path.exists(pt_path):
                print(f'PT file not found: {pt_path}')
                continue
                
            try:
                # 读取坐标和特征
                with h5py.File(h5_path, "r") as h5_file:
                    coords = np.array(h5_file['coords']).astype(int)
                
                features = torch.load(pt_path, map_location='cpu')
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                
                print(f'Processing {h5_fname}: coords shape {coords.shape}, features shape {features.shape}')
                
                # 创建图
                G = self.pt2graph_adjacentpatch(coords, features, radius=self.args.radius)
                
                # 保存图
                torch.save(G, graph_path)
                print(f'Graph saved: {graph_path}')
                
                # 验证保存的图
                loaded_G = torch.load(graph_path, map_location='cpu')
                print(f'Verification - Nodes: {loaded_G.x.shape[0]}, Edges: {loaded_G.edge_index.shape[1]}')
                
            except Exception as e:
                print(f"Error processing {h5_fname}: {e}")
                continue


class ConnectedComponents(Dataset):
    """构建基于连通组件的区域图"""
    def __init__(self, dataset_dir, args) -> None:
        super(ConnectedComponents, self).__init__()
        self.dataset_dir = dataset_dir
        self.patches_dir = os.path.join(dataset_dir, args.patches)
        self.pt_files_dir = os.path.join(dataset_dir, args.pt_files)
        self.graph_save_dir = os.path.join(dataset_dir, args.graph_files)
        self.args = args
        
        # 创建图保存目录
        os.makedirs(self.graph_save_dir, exist_ok=True)
        
        # 获取所有H5文件列表
        self.h5_files = [f for f in os.listdir(self.patches_dir) 
                        if f.endswith(".h5") and os.path.isfile(os.path.join(self.patches_dir, f))]
        self.h5_files.sort()
        
        print(f"Found {len(self.h5_files)} H5 files to process for region graphs")

    def get_feats(self, h5_fname):
        """从H5和PT文件获取特征和坐标"""
        h5_path = os.path.join(self.patches_dir, h5_fname)
        pt_fname = h5_fname.replace('.h5', '.pt')
        pt_path = os.path.join(self.pt_files_dir, pt_fname)
        
        # 读取坐标
        with h5py.File(h5_path, "r") as h5_file:
            coordinates = h5_file['coords'][()]
        
        # 读取特征
        features = torch.load(pt_path, map_location='cpu')
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        
        return features, coordinates

    def perform_kmeans_clustering(self, features):
        """执行K-means聚类"""
        kmeans_local = KMeans(n_clusters=self.args.num_clusters, random_state=42)
        clust_labels = kmeans_local.fit_predict(features)
        cent_coord = kmeans_local.cluster_centers_
        
        clusters = [[] for _ in range(self.args.num_clusters)]
        for i, label in enumerate(clust_labels):
            clusters[label].append(i)
        
        return clust_labels, cent_coord, clusters

    def get_binary_masks(self, cluster_labels, coordinates):
        """为每个聚类创建二值掩码"""
        min_x = np.min(coordinates[:, 0])
        min_y = np.min(coordinates[:, 1])
        max_x = np.max(coordinates[:, 0])
        max_y = np.max(coordinates[:, 1])

        height = max_y - min_y 
        width = max_x - min_x

        binary_masks = np.zeros((self.args.num_clusters, 
                               (height // self.args.patch_size) + 1, 
                               (width // self.args.patch_size) + 1))

        for i, (x, y) in enumerate(coordinates):
            adjusted_x = x - min_x 
            adjusted_y = y - min_y 

            patch_row = adjusted_y // self.args.patch_size
            patch_col = adjusted_x // self.args.patch_size

            cluster_id = cluster_labels[i]
            binary_masks[cluster_id, patch_row, patch_col] = 1

        return binary_masks, height, width

    def create_WSI(self, features, coordinates, height, width):
        """创建WSI特征矩阵"""
        min_x = np.min(coordinates[:, 0])
        min_y = np.min(coordinates[:, 1])

        n_patches_height = (height // self.args.patch_size) + 1 
        n_patches_width = (width // self.args.patch_size) + 1
        
        wsi_features = np.zeros((n_patches_height, n_patches_width, features.shape[1]))

        for i, feature_vector in enumerate(features):
            x, y = coordinates[i]
            adjusted_x = x - min_x 
            adjusted_y = y - min_y
            patch_row = adjusted_y // self.args.patch_size
            patch_col = adjusted_x // self.args.patch_size

            wsi_features[patch_row, patch_col] = feature_vector

        return wsi_features

    def connected_components(self, binary_mask, connectivity=2):
        """找到连通组件"""
        labeled_image, count = ski.measure.label(binary_mask, connectivity=connectivity, return_num=True)
        return labeled_image, count

    def join_labeled_imgs(self, labeled_images, height, width):
        """合并所有聚类的连通组件"""
        regions_img = None

        for i, labeled_img in enumerate(labeled_images):
            labeled_img_copy = labeled_img.copy()
            if regions_img is None:
                regions_img = labeled_img_copy
            else:
                biggest_indices = np.unravel_index(np.argmax(regions_img), regions_img.shape)
                max_value = regions_img[biggest_indices[0], biggest_indices[1]]
                
                labeled_img_copy[labeled_img_copy != 0] += max_value
                regions_img = np.maximum(regions_img, labeled_img_copy)
       
        return regions_img

    def get_cent_coords(self, region_img, threshold=0):
        """获取区域中心坐标"""
        cent_coords = {}
        nodes_to_remove = []
        region_properties = ski.measure.regionprops(region_img)
        nodes_to_remove.append(0)
        
        for region in region_properties:
            cent_coords[region.label] = region.centroid
            if region.area < threshold:
                nodes_to_remove.append(region.label)

        return cent_coords, nodes_to_remove

    def create_coords_mask(self, cent_coords, regions_img):
        """创建坐标掩码"""
        coords_mask = np.zeros((regions_img.shape[0], regions_img.shape[1], 3))
        
        for label, coords in cent_coords.items():
            region_indices = np.where(regions_img == label)
            row_indices = region_indices[0]
            col_indices = region_indices[1]
            coords_mask[row_indices, col_indices] = (coords[0], coords[1], 0)  
        return coords_mask

    def create_RAG(self, regions_img, coords_mask, nodes_to_remove):
        """创建区域邻接图"""
        try:
            # 尝试使用新的API
            rag = graph.rag_mean_color(coords_mask, regions_img, mode='distance')
        except AttributeError:
            # 如果新API不可用，使用旧API
            try:
                rag = graph.rag_mean_color(regions_img, coords_mask, mode='distance')
            except AttributeError:
                # 如果都不可用，创建简单的邻接图
                print("Warning: Using simplified RAG creation due to skimage API changes")
                rag = self._create_simple_rag(regions_img, nodes_to_remove)
        
        print(f"RAG - Nodes: {len(rag.nodes())}, Edges: {len(rag.edges())}")
        
        rag.remove_nodes_from(nodes_to_remove)
        return rag
    
    def _create_simple_rag(self, regions_img, nodes_to_remove):
        """创建简化的区域邻接图"""
        import networkx as nx
        
        # 获取所有区域标签
        unique_regions = np.unique(regions_img)
        unique_regions = unique_regions[unique_regions != 0]  # 排除背景
        
        # 创建空图
        G = nx.Graph()
        
        # 添加节点
        for region_id in unique_regions:
            G.add_node(region_id)
        
        # 添加边（基于空间邻接）
        for region_id in unique_regions:
            # 找到该区域的所有像素
            region_mask = (regions_img == region_id)
            
            # 找到边界像素
            from scipy import ndimage
            dilated = ndimage.binary_dilation(region_mask)
            boundary = dilated & ~region_mask
            
            # 找到相邻区域
            adjacent_regions = regions_img[boundary]
            adjacent_regions = adjacent_regions[adjacent_regions != 0]
            adjacent_regions = np.unique(adjacent_regions)
            
            # 添加边
            for adj_region in adjacent_regions:
                if adj_region != region_id and adj_region in unique_regions:
                    G.add_edge(region_id, adj_region, weight=1.0)
        
        return G

    def compute_region_features(self, regions_img, wsi_features):
        """计算区域特征"""
        regions = np.unique(regions_img)
        region_features = {}
        
        for region_id in regions:
            if region_id == 0:
                continue
            region_indices = np.where(regions_img == region_id)
            feature_vectors = wsi_features[region_indices]
            mean_feature = np.mean(feature_vectors, axis=0)
            region_features[region_id] = mean_feature

        return region_features

    def create_pyg_graph(self, region_features, rag):
        """创建PyG图"""
        edge_index = []
        edge_weights = []
        
        valid_nodes = {node: i for i, node in enumerate(rag.nodes())}
        
        node_features = []
        for node in rag.nodes():
            if node in region_features:
                node_features.append(region_features[node])
            else:
                raise ValueError(f"Node {node} in RAG is not in region_features")
        
        node_features_np = np.array(node_features)
        x = torch.tensor(node_features_np, dtype=torch.float)
        
        for (u, v, data) in rag.edges(data=True):
            if u in valid_nodes and v in valid_nodes:
                edge_index.append([valid_nodes[u], valid_nodes[v]])
                edge_weights.append(data['weight'])
        
        edge_index = np.array(edge_index).T 
        edge_index = torch.LongTensor(edge_index).to(device)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        edge_weights = np.array(edge_weights)  
        
        print(f'PyG Graph - Nodes: {x.shape[0]}, Edges: {edge_index.shape[1]}')

        data = geomData(x=x, edge_index=edge_index)
        return data, edge_index

    def create_region_graphs(self):
        """为所有文件创建区域图"""
        print('\n=== Creating region graphs ===\n')
        
        for i, h5_fname in enumerate(self.h5_files):
            print(f'Processing {i+1}/{len(self.h5_files)}: {h5_fname}')
            graph_fname = h5_fname.split('.')[0] + '.pt'
            graph_path = os.path.join(self.graph_save_dir, graph_fname)
            
            # 检查图文件是否已存在
            if os.path.exists(graph_path):
                print(f'Region graph already exists: {graph_fname}')
                continue
                
            try:
                print(f'Creating region graph for {h5_fname}')
                
                # 获取特征和坐标
                feats, coordinates = self.get_feats(h5_fname)
                
                # K-means聚类
                cluster_labels, local_centroids, clusters = self.perform_kmeans_clustering(feats)
                
                # 创建二值掩码
                binary_masks, height, width = self.get_binary_masks(cluster_labels, coordinates)
                
                # 创建WSI特征矩阵
                wsi_features = self.create_WSI(feats, coordinates, height, width)
                
                # 连通组件分析
                labeled_images = []
                for i, binary_mask in enumerate(binary_masks):
                    labeled_image, _ = self.connected_components(binary_mask)
                    labeled_images.append(labeled_image)

                # 合并连通组件
                regions_img = self.join_labeled_imgs(labeled_images, height, width)
                
                # 获取区域中心坐标
                cent_coords, nodes_to_remove = self.get_cent_coords(regions_img)
                
                # 创建坐标掩码
                coords_mask = self.create_coords_mask(cent_coords, regions_img)
                
                # 创建RAG
                rag = self.create_RAG(regions_img, coords_mask, nodes_to_remove)
                
                # 计算区域特征
                region_features = self.compute_region_features(regions_img, wsi_features)
                
                # 创建PyG图
                graph, edge_index = self.create_pyg_graph(region_features, rag)
                
                # 保存图
                torch.save(graph, graph_path)
                print(f'Region graph saved: {graph_path}')
                
            except Exception as e:
                print(f"Error processing {h5_fname}: {e}")
                continue


def main():
    global device
    parser = argparse.ArgumentParser(description='Build graphs for WSI patches')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/changhai_dataset',
                       help='Dataset directory containing patches and pt_files folders')
    parser.add_argument('--type_graph', type=str, default='patch',
                       choices=['patch', 'region', 'both'],
                       help='Type of graph to build: patch, region, or both')
    parser.add_argument('--num_clusters', type=int, default=9,
                       help='Number of clusters for K-means (region graphs only)')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Size of patches')
    parser.add_argument('--radius', type=int, default=1,
                       help='Radius for adjacent patch detection (patch graphs only)')
    parser.add_argument('--pt_files', type=str, default='pt_files')
    parser.add_argument('--patches', type=str, default='patches')
    parser.add_argument('--graph_files', type=str, default='graph_files')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Graph type: {args.type_graph}")
    print(f"Patch size: {args.patch_size}")
    
    if args.type_graph in ['patch', 'both']:
        print(f"Radius for adjacent patches: {args.radius}")
        patch_graphs = PatchGraphs(args.dataset_dir, args)
        patch_graphs.create_patch_graphs()
    
    if args.type_graph in ['region', 'both']:
        print(f"Number of clusters: {args.num_clusters}")
        region_graphs = ConnectedComponents(args.dataset_dir, args)
        region_graphs.create_region_graphs()
    
    print("Graph building completed!")


if __name__ == '__main__':
    main()
