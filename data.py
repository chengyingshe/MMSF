import torch
from torch.utils.data import Dataset
import glob
import os
import pandas as pd
import logging
from typing import Optional, Dict, Any
from torch_geometric.data import Data as PyGData


class PatchFeaturesDataset(Dataset):
    def __init__(self, patch_features_dir, csv_label_file):
        label_df = pd.read_csv(csv_label_file)
        self.patch_features_files = [os.path.join(patch_features_dir, filename) for filename in label_df.iloc[:, 0].values]
        self.labels = label_df.iloc[:, 1].values
        

    def __len__(self):
        return len(self.patch_features_files)
    
    def __getitem__(self, idx):
        patch_features = torch.load(self.patch_features_files[idx]).to(torch.float32)
        # Handle different data formats
        if patch_features.ndim == 3:
            patch_features = patch_features.squeeze(0)
        
        return {
            'patch_features': patch_features,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized patch features
    """
    # Separate patch features from other data
    patch_features = [item['patch_features'] for item in batch]
    other_data = {}
    
    # Get keys for other data (time, status, label, clinical_vector)
    for key in batch[0].keys():
        if key != 'patch_features':
            other_data[key] = [item[key] for item in batch]
    
    return {
        'patch_features': patch_features,  # Keep as list since sizes vary
        **other_data
    }


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for classification tasks, adapted from MultiTaskDataset.
    """
    def __init__(self, 
                 dataset_dir: str,
                 patch_features_dir_name: str = 'pt_files',
                 classification_file: str = 'clinical_data.csv',
                 id_column: str = 'ID',
                 classification_label_column: str = 'T',
                 id_mapping_file: str = 'id_mapping.csv',
                 cache_size: int = 100):
        """
        Initialize classification dataset.
        
        Args:
            dataset_dir: Root directory containing data files
            patch_features_dir_name: Directory name for patch features
            classification_file: CSV file with classification labels
            id_column: Column name for patient ID
            classification_label_column: Column name for classification label
            id_mapping_file: CSV file mapping patient IDs to feature files
            cache_size: Size of feature cache
        """
        self.dataset_dir = dataset_dir
        self.patch_features_dir = os.path.join(dataset_dir, patch_features_dir_name)
        self.id_column = id_column
        self.classification_label_column = classification_label_column
        
        # Load classification data
        classification_path = os.path.join(dataset_dir, classification_file)
        if not os.path.exists(classification_path):
            raise FileNotFoundError(f"Classification file not found: {classification_path}")
        
        self.classification_data = pd.read_csv(classification_path)
        
        # Check if label column exists
        if classification_label_column not in self.classification_data.columns:
            raise ValueError(f"Label column '{classification_label_column}' not found in classification file")
        
        # Create label mapping
        raw_values = self.classification_data[classification_label_column].dropna().astype(str)
        unique_vals = sorted(raw_values.unique().tolist())
        logging.info(f"Unique classification labels: {unique_vals}")
        self.class_label_to_index = {v: i for i, v in enumerate(unique_vals)}
        self.index_to_class_label = {i: v for v, i in self.class_label_to_index.items()}
        self.num_classes = len(unique_vals)
        
        # Load ID mapping
        mapping_file = os.path.join(dataset_dir, id_mapping_file)
        if os.path.exists(mapping_file):
            self.id_mapping = pd.read_csv(mapping_file)
            self.id_to_filename = dict(zip(self.id_mapping['csv_id'], self.id_mapping['pt_filename']))
        else:
            logging.warning(f"ID mapping file not found: {mapping_file}")
            self.id_to_filename = {}
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        # Feature cache
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []
        
        logging.info(f"ClassificationDataset initialized with {len(self.samples)} samples")
        logging.info(f"Number of classes: {self.num_classes}")
    
    def _build_sample_list(self):
        """Build list of valid samples with classification labels."""
        samples = []
        
        for _, row in self.classification_data.iterrows():
            patient_id = str(row[self.id_column])
            
            # Check if classification label is valid
            if pd.isna(row[self.classification_label_column]):
                continue
            
            raw_label = str(row[self.classification_label_column])
            if raw_label not in self.class_label_to_index:
                continue
            
            # Check if patch features exist
            if patient_id in self.id_to_filename:
                feature_file = self.id_to_filename[patient_id]
            else:
                feature_file = f"{patient_id}.pt"
            
            feature_path = os.path.join(self.patch_features_dir, feature_file)
            if not os.path.exists(feature_path):
                continue
            
            # Create sample record
            sample = {
                'patient_id': patient_id,
                'feature_path': feature_path,
                'label': self.class_label_to_index[raw_label]
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get sample by index."""
        sample = self.samples[idx]
        
        # Load patch features (with caching)
        if idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            patch_features = self._cache[idx]
        else:
            # Load from file
            patch_features = torch.load(sample['feature_path']).to(torch.float32)
            
            if patch_features.dim() > 2:
                patch_features = patch_features.squeeze(0)
            
            # Add to cache
            if len(self._cache) >= self.cache_size:
                # Remove least recently used
                oldest_idx = self._cache_order.pop(0)
                del self._cache[oldest_idx]
            
            self._cache[idx] = patch_features
            self._cache_order.append(idx)
        
        return {
            'patch_features': patch_features,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id']
        }


class SurvivalDataset(torch.utils.data.Dataset):
    """
    Dataset for survival tasks, aligned with the multitask project's expectations.
    CSV should contain columns: ID, OS, Status. An optional id_mapping.csv can map IDs to pt filenames.
    """
    def __init__(self,
                 dataset_dir: str,
                 patch_features_dir_name: str = 'pt_files',
                 survival_file: str = 'survival_data.csv',
                 id_column: str = 'ID',
                 time_column: str = 'OS',
                 event_column: str = 'Status',
                 id_mapping_file: str = 'id_mapping.csv',
                 cache_size: int = 100,
                 clinical_num_cols: Optional[list] = None,
                 clinical_cat_cols: Optional[list] = None,
                 clinical_norm: str = 'zscore'):
        self.dataset_dir = dataset_dir
        self.patch_features_dir = os.path.join(dataset_dir, patch_features_dir_name)
        self.id_column = id_column
        self.time_column = time_column
        self.event_column = event_column

        surv_path = os.path.join(dataset_dir, survival_file)
        if not os.path.exists(surv_path):
            raise FileNotFoundError(f"Survival file not found: {surv_path}")
        self.survival_df = pd.read_csv(surv_path)

        mapping_file = os.path.join(dataset_dir, id_mapping_file)
        if os.path.exists(mapping_file):
            mapping_df = pd.read_csv(mapping_file)
            self.id_to_filename = {str(k): v for k, v in zip(mapping_df['csv_id'], mapping_df['pt_filename'])}
        else:
            logging.warning(f"ID mapping file not found: {mapping_file}")
            self.id_to_filename = {}
        
        # Prepare clinical feature configuration
        self.clinical_norm = clinical_norm
        self.clinical_feature_names = []
        self.clinical_feature_types = []  # 'num' or 'cat'
        self.clinical_aux_dims = []       # 1 for cont, K for cat one-hot

        # If user provided explicit splits for cont/cat, honor them; else infer from dtype
        # Use list comprehension to preserve order (instead of set which may change order)
        inferred_num = [c for c in clinical_num_cols if c in self.survival_df.columns] if clinical_num_cols else []
        inferred_cat = [c for c in clinical_cat_cols if c in self.survival_df.columns] if clinical_cat_cols else []

        # Build encoders/statistics with missing value imputation
        self._num_stats: Dict[str, Dict[str, float]] = {}
        
        # preprocessing numerical features
        for c in inferred_num:
            series_all = pd.to_numeric(self.survival_df[c], errors='coerce')
            mean_val = float(series_all.mean(skipna=True)) if not pd.isna(series_all.mean(skipna=True)) else 0.0
            series = series_all.fillna(mean_val)
            if len(series) == 0:
                mean, std = 0.0, 1.0
                min_v, max_v = 0.0, 1.0
            else:
                mean = float(series.mean())
                std_calc = float(series.std())
                std = std_calc if std_calc != 0 else 1.0
                min_v = float(series.min())
                max_v = float(series.max())
                if max_v == min_v:
                    max_v = min_v + 1.0
            self._num_stats[c] = {'mean': mean, 'std': std, 'min': min_v, 'max': max_v}
            self.clinical_feature_names.append(c)
            self.clinical_feature_types.append('num')
            self.clinical_aux_dims.append(1)

        self._cat_maps: Dict[str, Dict[str, int]] = {}
        
        # preprocessing categorical features
        for c in inferred_cat:
            # mode imputation for categorical
            col = self.survival_df[c]
            mode_val = None
            try:
                mode_series = col.mode(dropna=True)
                if len(mode_series) > 0:
                    mode_val = mode_series.iloc[0]
            except Exception:
                mode_val = None
            if pd.isna(mode_val):
                mode_val = 'NA'
            col_filled = col.fillna(mode_val).astype(str)
            raw_vals = col_filled.astype(str)
            classes = sorted(raw_vals.unique().tolist())
            if len(classes) == 0:
                classes = ['NA']
            value_to_index = {v: i for i, v in enumerate(classes)}
            self._cat_maps[c] = value_to_index
            self.clinical_feature_names.append(c)
            self.clinical_feature_types.append('cat')
            self.clinical_aux_dims.append(len(classes))

        self.samples = []
        self.survival_df[self.id_column] = self.survival_df[self.id_column].astype(str)
        for _, row in self.survival_df.iterrows():
            pid = row[self.id_column]
            if pd.isna(pid):
                continue
            # map to feature path
            fname = self.id_to_filename.get(pid, f"{pid}.pt")
            fpath = os.path.join(self.patch_features_dir, fname)
            if not os.path.exists(fpath):
                logging.debug(f"Missing pt file for ID {pid}: {fpath}")
                continue
            time_val = row[self.time_column]
            event_val = row[self.event_column]
            if pd.isna(time_val) or pd.isna(event_val):
                logging.debug(f"Missing time or event data for ID {pid}")
                continue
            # Build clinical vector for this row
            clinical_vec = []
            valid = True
            for fname, ftype, aux in zip(self.clinical_feature_names, self.clinical_feature_types, self.clinical_aux_dims):
                if ftype == 'num':
                    v = row.get(fname)
                    try:
                        v = float(v)
                    except Exception:
                        v = None
                    if v is None or pd.isna(v):
                        v = self._num_stats[fname]['mean']
                    if self.clinical_norm == 'zscore':
                        v = (v - self._num_stats[fname]['mean']) / self._num_stats[fname]['std']
                    else:
                        # minmax
                        v = (v - self._num_stats[fname]['min']) / (self._num_stats[fname]['max'] - self._num_stats[fname]['min'])
                    clinical_vec.append(float(v))
                else:
                    raw = row.get(fname)
                    # fill with mode if missing
                    if pd.isna(raw):
                        # choose first class as mode placeholder
                        mode_idx = 0
                        for k, idx in self._cat_maps[fname].items():
                            if idx == 0:
                                raw = k
                                break
                    raw = str(raw)
                    idx = self._cat_maps[fname].get(raw, None)
                    onehot = [0.0] * aux
                    if idx is not None:
                        onehot[idx] = 1.0
                    clinical_vec.extend(onehot)

            self.samples.append({
                'patient_id': pid,
                'feature_path': fpath,
                'time': float(time_val),
                'status': int(event_val),
                'clinical_vector': clinical_vec
            })

        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []
        
        logging.info(f'Clinical aux dims: {self.clinical_aux_dims}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # cache
        if idx in self._cache:
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            patch_features = self._cache[idx]
        else:
            patch_features = torch.load(sample['feature_path']).to(torch.float32)
            if patch_features.dim() > 2:
                patch_features = patch_features.squeeze(0)
            if len(self._cache) >= self.cache_size:
                oldest = self._cache_order.pop(0) if self._cache_order else None
                if oldest is not None and oldest in self._cache:
                    del self._cache[oldest]
            self._cache[idx] = patch_features
            self._cache_order.append(idx)

        return {
            'patient_id': sample['patient_id'],
            'patch_features': patch_features,
            'time': torch.tensor(sample['time'], dtype=torch.float32),
            'status': torch.tensor(sample['status'], dtype=torch.long),
            'clinical_vector': torch.tensor(sample['clinical_vector'], dtype=torch.float32)
        }


class GraphClassificationDataset(ClassificationDataset):
    """
    Classification dataset with optional graph data loading.
    Expects graph files saved under `graph_files/` (or configurable) as PyG Data objects
    keyed by the same base filename as patch features (via `id_mapping.csv` if present).
    """
    def __init__(self,
                 dataset_dir: str,
                 patch_features_dir_name: str = 'pt_files',
                 graph_features_dir_name: str = 'graph_files',
                 classification_file: str = 'clinical_data.csv',
                 id_column: str = 'ID',
                 classification_label_column: str = 'T',
                 id_mapping_file: str = 'id_mapping.csv',
                 use_graph_features: bool = True,
                 graph_type: str = 'patch',  # 'patch' or 'region'
                 cache_size: int = 100):

        super().__init__(
            dataset_dir=dataset_dir,
            patch_features_dir_name=patch_features_dir_name,
            classification_file=classification_file,
            id_column=id_column,
            classification_label_column=classification_label_column,
            id_mapping_file=id_mapping_file,
            cache_size=cache_size,
        )

        self.use_graph_features = bool(use_graph_features)
        self.graph_type = graph_type
        self.graph_features_dir = os.path.join(dataset_dir, graph_features_dir_name)
        
        mapping_file = os.path.join(dataset_dir, id_mapping_file)
        if os.path.exists(mapping_file):
            mapping_df = pd.read_csv(mapping_file)
            self.id_to_filename = {str(k): v for k, v in zip(mapping_df['csv_id'], mapping_df['pt_filename'])}
        else:
            logging.warning(f"ID mapping file not found: {mapping_file}")
            self.id_to_filename = {}

        if self.use_graph_features:
            if PyGData is None:
                logging.warning("torch_geometric is not available; disabling graph features.")
                self.use_graph_features = False
            elif not os.path.exists(self.graph_features_dir):
                logging.warning(f"Graph features directory not found: {self.graph_features_dir}. Disabling graph features.")
                self.use_graph_features = False

        if self.use_graph_features:
            # Filter to samples that have matching graph data
            filtered = []
            for s in self.samples:
                pid = s['patient_id']
                # Prefer mapped filename if available
                if hasattr(self, 'id_to_filename') and pid in self.id_to_filename:
                    gname = self.id_to_filename[pid]
                else:
                    gname = f"{pid}.pt"
                gpath = os.path.join(self.graph_features_dir, gname)
                if os.path.exists(gpath):
                    filtered.append(s)
                else:
                    logging.debug(f"Missing graph for patient {pid}: {gpath}")
            
            logging.info(f"GraphClassificationDataset: {len(filtered)}/{len(self.samples)} samples have graph data")
            self.samples = filtered

    def _load_graph(self, patient_id: str) -> Optional[Any]:
        if not self.use_graph_features:
            return None
        if hasattr(self, 'id_to_filename') and patient_id in self.id_to_filename:
            gname = self.id_to_filename[patient_id]
        else:
            gname = f"{patient_id}.pt"
        gpath = os.path.join(self.graph_features_dir, gname)

        try:
            graph_data = torch.load(gpath, map_location='cpu')
            # Basic validation if PyG is available
            if PyGData is not None and not isinstance(graph_data, PyGData):
                logging.error("Loaded graph object is not a torch_geometric.data.Data instance")
                return None
            if not hasattr(graph_data, 'x') or not hasattr(graph_data, 'edge_index'):
                logging.error("Graph data missing x or edge_index")
                return None
            return graph_data
        except Exception as e:  # pragma: no cover
            logging.error(f"Failed to load graph for {patient_id}: {e}")
            return None

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.use_graph_features:
            item['graph_data'] = self._load_graph(item['patient_id'])
        else:
            item['graph_data'] = None
        return item


class GraphSurvivalDataset(SurvivalDataset):
    """
    Survival dataset with optional graph data loading.
    Expects graph files saved under `graph_files/` (or configurable) as PyG Data objects
    keyed by the same base filename as patch features (via `id_mapping.csv` if present).
    """
    def __init__(self,
                 dataset_dir: str,
                 patch_features_dir_name: str = 'pt_files',
                 graph_features_dir_name: str = 'graph_files',
                 survival_file: str = 'survival_data.csv',
                 id_column: str = 'ID',
                 time_column: str = 'OS',
                 event_column: str = 'Status',
                 id_mapping_file: str = 'id_mapping.csv',
                 use_graph_features: bool = True,
                 graph_type: str = 'patch',  # 'patch' or 'region'
                 cache_size: int = 100,
                 clinical_num_cols: Optional[list] = None,
                 clinical_cat_cols: Optional[list] = None,
                 clinical_norm: str = 'zscore'):

        super().__init__(
            dataset_dir=dataset_dir,
            patch_features_dir_name=patch_features_dir_name,
            survival_file=survival_file,
            id_column=id_column,
            time_column=time_column,
            event_column=event_column,
            id_mapping_file=id_mapping_file,
            cache_size=cache_size,
            clinical_num_cols=clinical_num_cols,
            clinical_cat_cols=clinical_cat_cols,
            clinical_norm=clinical_norm,
        )

        self.use_graph_features = bool(use_graph_features)
        self.graph_type = graph_type
        self.graph_features_dir = os.path.join(dataset_dir, graph_features_dir_name)
        
        mapping_file = os.path.join(dataset_dir, id_mapping_file)
        if os.path.exists(mapping_file):
            mapping_df = pd.read_csv(mapping_file)
            self.id_to_filename = {str(k): v for k, v in zip(mapping_df['csv_id'], mapping_df['pt_filename'])}
        else:
            logging.warning(f"ID mapping file not found: {mapping_file}")
            self.id_to_filename = {}

        if self.use_graph_features:
            if PyGData is None:
                logging.warning("torch_geometric is not available; disabling graph features.")
                self.use_graph_features = False
            elif not os.path.exists(self.graph_features_dir):
                logging.warning(f"Graph features directory not found: {self.graph_features_dir}. Disabling graph features.")
                self.use_graph_features = False

        if self.use_graph_features:
            # Filter to samples that have matching graph data
            filtered = []
            for s in self.samples:
                pid = s['patient_id']
                # Prefer mapped filename if available
                if hasattr(self, 'id_to_filename') and pid in self.id_to_filename:
                    gname = self.id_to_filename[pid]
                else:
                    gname = f"{pid}.pt"
                gpath = os.path.join(self.graph_features_dir, gname)
                if os.path.exists(gpath):
                    filtered.append(s)
                else:
                    logging.debug(f"Missing graph for patient {pid}: {gpath}")
            
            logging.info(f"GraphSurvivalDataset: {len(filtered)}/{len(self.samples)} samples have graph data")
            self.samples = filtered

    def _load_graph(self, patient_id: str) -> Optional[Any]:
        if not self.use_graph_features:
            return None
        if hasattr(self, 'id_to_filename') and patient_id in self.id_to_filename:
            gname = self.id_to_filename[patient_id]
        else:
            gname = f"{patient_id}.pt"
        gpath = os.path.join(self.graph_features_dir, gname)

        try:
            graph_data = torch.load(gpath, map_location='cpu')
            # Basic validation if PyG is available
            if PyGData is not None and not isinstance(graph_data, PyGData):
                logging.error("Loaded graph object is not a torch_geometric.data.Data instance")
                return None
            if not hasattr(graph_data, 'x') or not hasattr(graph_data, 'edge_index'):
                logging.error("Graph data missing x or edge_index")
                return None
            return graph_data
        except Exception as e:  # pragma: no cover
            logging.error(f"Failed to load graph for {patient_id}: {e}")
            return None

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.use_graph_features:
            item['graph_data'] = self._load_graph(item['patient_id'])
        else:
            item['graph_data'] = None
        return item


def collate_fn_with_graph(batch: list) -> Dict[str, Any]:
    """
    Collate function that preserves variable-sized bags and attaches per-sample graph objects.
    """
    out = collate_fn(batch)
    graphs = []
    for b in batch:
        graphs.append(b.get('graph_data'))
    out['graph_data'] = graphs
    return out

