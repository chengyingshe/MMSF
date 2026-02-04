import copy
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FCLayer(nn.Module):
    """Fully connected layer for instance-level classification."""
    
    def __init__(self, in_size: int, out_size: int = 1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            feats: Input features [N, D]
            
        Returns:
            Tuple of (features, predictions)
        """
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    """Instance-level classifier base class."""
    
    def __init__(self, feature_extractor: nn.Module, feature_size: int, output_class: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [N, D]
            
        Returns:
            Tuple of (features, classification scores)
        """
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class IntelligentPatchSelector(nn.Module):
    """
    Intelligent patch selection module for selecting important patches
    from a bag of instances using different strategies.
    """
    
    def __init__(self, selection_strategy: str = 'hybrid', 
                 diversity_weight: float = 0.3, 
                 uncertainty_weight: float = 0.3):
        super().__init__()
        self.selection_strategy = selection_strategy
        self.diversity_weight = diversity_weight
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, x: Tensor, c: Tensor, big_lambda: int, 
                top_big_lambda_share: float = 0.7, 
                random_patch_share: float = 0.3) -> Tuple[Tensor, Tensor]:
        """
        Select patches based on different strategies.
        
        Args:
            x: Input features [B, N, D]
            c: Instance classification scores [B, N, C]
            big_lambda: Total number of patches to select
            top_big_lambda_share: Ratio of top patches to select
            random_patch_share: Ratio of random patches to select
        
        Returns:
            Tuple of (selected_indices, attention_weights)
        """
        B, N, D = x.shape
        B, N, C = c.shape
        
        if self.selection_strategy == 'hybrid':
            return self._hybrid_selection(x, c, big_lambda)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
    
    
    def _hybrid_selection(self, x: Tensor, c: Tensor, big_lambda: int) -> Tuple[Tensor, Tensor]:
        """
        Hybrid selection strategy combining relevance, diversity, and uncertainty.
        """
        B, N, _ = x.shape
        
        # Ensure big_lambda does not exceed available patch count
        big_lambda = min(big_lambda, N)
        
        # Relevance scores (based on instance classification scores)
        if c.shape[-1] == 1:
            relevance_scores = c.squeeze(-1)  # [B, N]
        else:
            relevance_scores = c.max(dim=-1)[0]  # [B, N] - use max class score
        
        # Diversity scores (based on feature distances)
        diversity_scores = self._compute_diversity_scores(x)  # [B, N]
        
        # Uncertainty scores (based on prediction entropy)
        uncertainty_scores = self._compute_uncertainty_scores(c)  # [B, N]
        
        # Combine scores
        combined_scores = (
            relevance_scores + 
            self.diversity_weight * diversity_scores +
            self.uncertainty_weight * uncertainty_scores
        )
        
        # Select top-k (ensure not exceeding available count)
        _, selected_indices = torch.topk(combined_scores, big_lambda, dim=1)
        
        # Compute attention weights
        attention_weights = F.softmax(combined_scores, dim=1)
        
        return selected_indices, attention_weights
    
    
    def _compute_diversity_scores(self, x: Tensor) -> Tensor:
        """Compute diversity scores based on feature distances."""
        B, N, D = x.shape
        
        # Compute average distance of each patch to all other patches
        # Memory-efficient computation without storing full similarity matrix
        x_norm = F.normalize(x, p=2, dim=-1)  # [B, N, D] - L2 normalization
        
        # Compute sum of all normalized features
        sum_all = x_norm.sum(dim=1, keepdim=True)  # [B, 1, D]
        
        # For each patch i, compute sum of similarities to all patches
        # similarity_sum[i] = sum_j dot(x_norm[i], x_norm[j])
        similarity_sum = torch.bmm(x_norm, sum_all.transpose(1, 2)).squeeze(-1)  # [B, N]
        
        # Exclude self-similarity (which is always 1 after normalization)
        # avg_similarity[i] = (similarity_sum[i] - 1) / (N - 1)
        avg_similarity = (similarity_sum - 1) / (N - 1)  # exclude self
        
        # Diversity = 1 - average similarity
        diversity_scores = 1 - avg_similarity
        
        return diversity_scores
    
    def _compute_uncertainty_scores(self, c: Tensor) -> Tensor:
        """Compute uncertainty scores based on prediction entropy."""
        # Entropy-based uncertainty from prediction probabilities
        if c.shape[-1] == 1:
            # For binary classification, create [prob, 1-prob] form
            probs = torch.sigmoid(c.squeeze(-1))  # [B, N]
            probs_full = torch.stack([1-probs, probs], dim=-1)  # [B, N, 2]
        else:
            probs_full = F.softmax(c, dim=-1)  # [B, N, C]
        
        uncertainty_scores = -torch.sum(probs_full * torch.log(probs_full + 1e-8), dim=-1)  # [B, N]
        return uncertainty_scores


class BaseBClassifier(nn.Module):
    """
    Base bag-level classifier that provides common functionality
    for different MIL architectures.
    """
    
    def __init__(self, input_size: int, 
                 output_class: int, 
                 big_lambda: int = 64,
                 selection_strategy: str = 'aps',   # top-k, random-k, aps
                 use_intelligent_selector: bool = True
                 ):
        super().__init__()
        
        # Validate selection strategy
        valid_strategies = ['random-k', 'top-k', 'aps']
        if selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid selection_strategy: {selection_strategy}. Must be one of {valid_strategies}")
        
        # Save parameters
        self.input_size = input_size
        self.output_class = output_class
        self.big_lambda = big_lambda
        self.selection_strategy = selection_strategy

        # Intelligent patch selector
        self.use_intelligent_selector = use_intelligent_selector
        if use_intelligent_selector and selection_strategy == 'aps':
            self.patch_selector = IntelligentPatchSelector(selection_strategy='hybrid')

    def _extract_selected_features(self, x: Tensor, selected_indices: Tensor, 
                                 big_lambda: int) -> Tensor:
        """
        Extract features based on selected indices.
        
        Args:
            x: Input features [1, N, D]
            selected_indices: Selected indices [big_lambda] or [1, big_lambda]
            big_lambda: Number of selected patches
        
        Returns:
            Extracted features [1, big_lambda, D]
        """
        B, N, D = x.shape
        device = x.device
        
        # Handle selected_indices dimensions
        if selected_indices.dim() == 2:
            selected_indices = selected_indices.squeeze(0)  # [big_lambda]
        
        # Ensure big_lambda does not exceed actual selectable count
        actual_big_lambda = min(big_lambda, len(selected_indices), N)
        
        # Extract selected features
        valid_indices = selected_indices[selected_indices >= 0]
        valid_indices = valid_indices[valid_indices < N]  # ensure indices are in valid range
        
        # Further limit valid index count
        if len(valid_indices) > actual_big_lambda:
            valid_indices = valid_indices[:actual_big_lambda]
        
        # Create output tensor
        m_feats = torch.zeros(1, actual_big_lambda, D, device=device)
        
        if len(valid_indices) > 0:
            selected_feats = x[0, valid_indices]  # [len(valid_indices), D]
            m_feats[0, :len(valid_indices)] = selected_feats
        
        return m_feats

    def _random_selection(self, N: int, k: int, device: torch.device) -> Tensor:
        """
        Randomly select k patches from N available patches.
        
        Args:
            N: Total number of available patches
            k: Number of patches to select
            device: Device to create tensor on
            
        Returns:
            Selected indices tensor [k]
        """
        # Ensure k doesn't exceed N
        k = min(k, N)
        
        # Generate random indices
        all_indices = torch.randperm(N, device=device)
        selected_indices = all_indices[:k]
        
        return selected_indices

    def _apply_patch_selection(self, feats: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply patch selection strategy.
        
        Args:
            feats: Input features [N, D]
            c: Instance classification scores [N, C]
            
        Returns:
            Tuple of (selected_features, attention_weights)
        """
        device = feats.device
        N, D = feats.shape
        
        # Dynamic adjustment of big_lambda
        effective_big_lambda = min(self.big_lambda, N)

        # Expand dimensions for batch processing (add batch dimension)
        feats = feats.unsqueeze(0)  # [1, N, D]
        c_expanded = c.unsqueeze(0)  # [1, N, C]

        # Patch selection based on strategy
        if self.selection_strategy == 'random-k':
            # Random selection strategy
            selected_indices = self._random_selection(N, effective_big_lambda, device)
            attention_weights = torch.ones(N, device=device) / N  # Uniform attention
        elif self.selection_strategy == 'top-k':
            # Simple top-k selection strategy
            if c.shape[-1] == 1:
                c_for_sort = c.squeeze(-1)  # [N]
            else:
                c_for_sort = c.max(dim=-1)[0]  # [N] - use max class score
            
            _, m_indices = torch.sort(c_for_sort, 0, descending=True)
            selected_indices = m_indices[:effective_big_lambda]
            attention_weights = F.softmax(c_for_sort, dim=0)
        elif self.selection_strategy == 'aps' and self.use_intelligent_selector:
            # Use IntelligentPatchSelector with 'aps' strategy (hybrid approach)
            selected_indices, attention_weights = self.patch_selector(feats, c_expanded, effective_big_lambda)
            selected_indices = selected_indices.squeeze(0)  # [effective_big_lambda]
            attention_weights = attention_weights.squeeze(0)  # [N]
        else:
            # Fallback to simple top-k selection strategy
            if c.shape[-1] == 1:
                c_for_sort = c.squeeze(-1)  # [N]
            else:
                c_for_sort = c.max(dim=-1)[0]  # [N] - use max class score
            
            _, m_indices = torch.sort(c_for_sort, 0, descending=True)
            selected_indices = m_indices[:effective_big_lambda]
            attention_weights = F.softmax(c_for_sort, dim=0)

        # Extract selected features
        m_feats = self._extract_selected_features(feats, selected_indices, effective_big_lambda)
        
        return m_feats, attention_weights


class MILNet(nn.Module):    
    def __init__(self, i_classifier: nn.Module, b_classifier: nn.Module):
        super().__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """        
        Args:
            x: Input features [N, D]
        
        Returns:
            Tuple: (instance_predictions, bag_prediction, attention_weights, bag_representation)
        """
        # Instance-level classification
        feats, classes = self.i_classifier(x)
        # Bag-level classification
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
