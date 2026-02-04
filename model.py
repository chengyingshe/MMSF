import logging
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from torch import Tensor
from common import FCLayer
from mamba_mil import BClassifier as MambaBagClassifier
from graph_encoders import GCNEncoder, GATEncoder, GraphMambaEncoder
from clinical_encoders import ClinicalEmbedding

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeatureFusion(nn.Module):
    """
    Feature fusion module for combining two feature vectors.
    Supports four fusion strategies: none, linear, se, cross_attention.
    """
    
    def __init__(self, input_dim: int, fuse_type: str = 'none', hidden_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.fuse_type = fuse_type
        self.hidden_dim = hidden_dim or input_dim // 2
        
        if fuse_type == 'linear':
            # Linear projection after concatenation
            self.fuser = nn.Linear(input_dim, input_dim)
        elif fuse_type == 'se':
            # SE attention mechanism
            self.fuser = SELayer(input_dim, reduction=max(1, input_dim // 4))
            self.norm = nn.LayerNorm(input_dim)
        else:  # 'none'
            # Direct concatenation, no additional processing
            self.fuser = nn.Identity()
    
    def forward(self, feat1: Tensor, feat2: Tensor) -> Tensor:
        """
        Fuse two feature vectors using specified fusion strategy.
        
        Args:
            feat1: First feature vector [N, input_dim//2]
            feat2: Second feature vector [N, input_dim//2]
            
        Returns:
            Fused features [N, input_dim]
        """
        fused = torch.cat([feat1, feat2], dim=1)
        
        if self.fuse_type == 'se':
            # Reshape for SELayer (expects 4D input)
            original_shape = fused.shape
            fused = self.norm(fused)
            fused = fused.view(fused.size(0), fused.size(1), 1, 1)
            fused = self.fuser(fused)
            fused = fused.view(original_shape)
        else:
            fused = self.fuser(fused)
            
        return fused


class Network(nn.Module):

    def __init__(
        self,
        feats_size: int,
        output_class: int = 1,
        dropout: float = 0.1,
        big_lambda: int = 64,
        selection_strategy: str = 'aps',  # 'random-k' | 'top-k' | 'aps'
        # task options
        task: str = 'survival',  # 'survival' | 'classification'
        num_classes: int = 2,  # for classification task
        # mamba options
        mamba_depth: int = 8,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        # graph options
        use_graph: bool = False,
        graph_model: str = 'gat',  # 'gcn' | 'gat'
        graph_hidden: int = 256,
        graph_out: int = 256,
        graph_dropout: float = 0.1,
        fuse_type: str = 'se',  # 'none' | 'linear' | 'se' | 'cross_attention'
        # clinical options
        use_clinical: bool = False,
        clinical_hidden: int = 128,
        clinical_aux_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        # Task configs
        self.task = task
        self.num_classes = num_classes

        self.use_graph = use_graph
        self.graph_model = graph_model
        self.graph_hidden = graph_hidden
        self.graph_out = graph_out
        self.graph_dropout = graph_dropout
        self.fuse_type = fuse_type

        # Clinical configs
        self.use_clinical = use_clinical
        self.clinical_hidden = clinical_hidden
        self.clinical_aux_dims = clinical_aux_dims

        # Calculate feature dimensions for patch-level fusion
        patch_fusion_input_dim = feats_size
        if self.use_graph:
            patch_fusion_input_dim += self.graph_out
        
        # Calculate effective patch features dimension after fusion
        if self.use_graph:
            # if self.fuse_type == 'linear':
            #     effective_patch_feats = feats_size  # map back to base dim
            # else:  # 'none', 'se', 'cross_attention'
            effective_patch_feats = patch_fusion_input_dim
        else:
            effective_patch_feats = feats_size

        # Instance classifier on effective patch features
        self.i_classifier = FCLayer(in_size=effective_patch_feats, out_size=output_class)

        # Bag-level Mamba classifier on effective patch features
        self.b_classifier = MambaBagClassifier(
            input_size=effective_patch_feats,
            output_class=output_class,
            mamba_depth=mamba_depth,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            big_lambda=big_lambda,
            selection_strategy=selection_strategy,
            use_intelligent_selector=True,
            dropout=dropout,
        )

        # Calculate feature dimensions for instance-level fusion
        # bag_representation dimension is the same as effective_patch_feats
        instance_fusion_input_dim = effective_patch_feats
        if self.use_clinical:
            instance_fusion_input_dim += self.clinical_hidden
        
        # Calculate final feature dimension after instance-level fusion
        # if self.fuse_type == 'linear':
        #     final_feature_dim = effective_patch_feats  # map back to patch features dim
        # else:  # 'none', 'se', 'cross_attention'
        final_feature_dim = instance_fusion_input_dim

        # Feature fusion modules
        self.graph_feature_fusion = None
        if self.use_graph:
            # input_dim is the total dimension after concatenation
            graph_fusion_input_dim = feats_size + self.graph_out
            self.graph_feature_fusion = FeatureFusion(
                input_dim=graph_fusion_input_dim,
                fuse_type=fuse_type
            )
        
        self.clinical_feature_fusion = None
        if self.use_clinical:
            # input_dim is the total dimension after concatenation
            clinical_fusion_input_dim = effective_patch_feats + self.clinical_hidden
            self.clinical_feature_fusion = FeatureFusion(
                input_dim=clinical_fusion_input_dim,
                fuse_type=fuse_type
            )

        # Task-specific heads
        if self.task == 'survival':
            self.head = nn.Sequential(
                nn.Linear(final_feature_dim, 1),
                nn.Sigmoid()
            )
        elif self.task == 'classification':
            self.head = nn.Sequential(
                nn.LayerNorm(final_feature_dim),
                nn.Dropout(dropout),
                nn.Linear(final_feature_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown task: {self.task}. Must be 'survival' or 'classification'")

        # Optional graph encoder
        self.graph_encoder = None
        if self.use_graph:
            if self.graph_model == 'gcn':
                self.graph_encoder = GCNEncoder(
                    in_channels=feats_size,
                    hidden_channels=graph_hidden,
                    out_channels=graph_out,
                    dropout=graph_dropout,
                )
            elif self.graph_model == 'gat':
                self.graph_encoder = GATEncoder(
                    in_channels=feats_size,
                    hidden_channels=graph_hidden,
                    out_channels=graph_out,
                    heads=4,
                    dropout=graph_dropout,
                )
            else:
                raise ValueError(f"Unknown graph model: {self.graph_model}. Must be 'gcn' or 'gat'")

        # Optional ClinicalEmbedding for clinical features
        self.clinical_embedding = None
        if self.use_clinical and self.clinical_aux_dims is not None and len(self.clinical_aux_dims) > 0:
            self.clinical_embedding = ClinicalEmbedding(hidden_dim=self.clinical_hidden, n_aux_classes=self.clinical_aux_dims)

    @staticmethod
    def _extract_graph_tensors(graph_data: Any, device: torch.device) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if graph_data is None:
            return None, None, None
        x = getattr(graph_data, 'x', None)
        edge_index = getattr(graph_data, 'edge_index', None)
        batch = getattr(graph_data, 'batch', None)
        if x is not None:
            x = x.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if batch is not None:
            batch = batch.to(device)
        return x, edge_index, batch


    def forward(self, patch_features: Tensor, graph_data: Optional[Any] = None, clinical_vector: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Args:
            patch_features: Tensor [N, F]
            graph_data: Optional torch_geometric.data.Data with attributes x, edge_index, batch
            clinical_vector: Optional Tensor [clinical_dim] for clinical features

        Returns dict with:
            - instance_predictions: [N, C]
            - bag_prediction: [1, C] - task-specific output (risk score or classification logits)
            - attention_weights: [N, C]
            - bag_representation: [1, D] - final fused features
            - node_embeddings: optional [N, G]
            - graph_embedding: optional [1, G]
        """
        device = patch_features.device

        # Step 1: Extract graph embeddings if available
        node_emb = None
        graph_emb = None
        if self.use_graph and graph_data is not None and self.graph_encoder is not None:
            x, edge_index, batch = self._extract_graph_tensors(graph_data, device)
            if x is not None:
                try:
                    if edge_index is None:
                        node_emb, graph_emb = self.graph_encoder(x, edge_index)
                    else:
                        try:
                            node_emb, graph_emb = self.graph_encoder(x, edge_index, batch)
                        except TypeError:
                            node_emb, graph_emb = self.graph_encoder(x, edge_index)
                except Exception as e:
                    logging.warning(f"Graph encoder failed ({type(self.graph_encoder).__name__}): {e}; skipping graph fusion")
                    node_emb, graph_emb = None, None

        # Step 2: Extract clinical embeddings if available
        clinical_emb = None
        clinical_recon = None
        if self.use_clinical and self.clinical_embedding is not None and clinical_vector is not None:
            try:
                if clinical_vector.dim() == 1:
                    clinical_vector = clinical_vector.unsqueeze(0)
                clinical_emb, clinical_preds = self.clinical_embedding(clinical_vector)
                clinical_recon = clinical_preds
            except Exception as e:
                logging.warning(f"clinical_embedding forward failed: {e}")
                clinical_emb = None
                clinical_recon = None

        # Step 3: Patch-level fusion (patch_features + node_emb)
        if self.use_graph and self.graph_feature_fusion is not None:
            if node_emb is not None and node_emb.size(0) == patch_features.size(0):
                fused_patch_feats = self.graph_feature_fusion(patch_features, node_emb)
            else:
                if node_emb is not None and node_emb.size(0) != patch_features.size(0):
                    logging.warning(f"Graph/feature count mismatch: {node_emb.size(0)} vs {patch_features.size(0)}; using zeros for node_emb")
                # fill zeros to keep concat dim consistent
                zeros_graph = torch.zeros(patch_features.size(0), self.graph_out, device=patch_features.device, dtype=patch_features.dtype)
                fused_patch_feats = self.graph_feature_fusion(patch_features, zeros_graph)
        else:
            fused_patch_feats = patch_features

        # Step 4: Instance-level classification and bag-level processing
        feats, classes = self.i_classifier(fused_patch_feats)
        feats = feats.contiguous()
        bag_representation, attention_weights = self.b_classifier(feats, classes)

        # Step 5: Instance-level fusion (bag_representation + clinical_emb)
        if self.use_clinical and self.clinical_feature_fusion is not None and clinical_emb is not None:
            # clinical_emb is [1, clinical_hidden], bag_representation is [1, effective_patch_feats]
            final_features = self.clinical_feature_fusion(bag_representation, clinical_emb)
        else:
            final_features = bag_representation

        # Step 6: Apply task-specific head
        bag_prediction = self.head(final_features)

        # Prepare output dictionary
        out: Dict[str, Tensor] = {
            'patch_scores': classes,
            'attention_weights': attention_weights,
            'bag_representation': final_features,  # Final fused features (instance-level features)
            'bag_prediction': bag_prediction,  # risk score | classification logits
        }
        
        # Add optional outputs
        if node_emb is not None:
            out['node_embeddings'] = node_emb
        if graph_emb is not None:
            out['graph_embedding'] = graph_emb
        if clinical_emb is not None:
            out['clinical_embedding'] = clinical_emb
        if clinical_recon is not None:
            out['clinical_recon'] = clinical_recon
            
        return out


def initialize_weights(module: nn.Module) -> None:
    """Initialize model weights using standard methods."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)