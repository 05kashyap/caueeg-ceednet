import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from collections import OrderedDict

from .activation import get_activation_class


class PatchEmbedding1D(nn.Module):
    """Convert 1D EEG signals to patches similar to ViT."""
    
    def __init__(self, seq_length: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding using 1D convolution
        self.proj = nn.Conv1d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, seq_length)
        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        B, C, L = x.shape
        assert L == self.seq_length, f"Input length {L} doesn't match expected {self.seq_length}"
        
        # Apply patch embedding
        x = self.proj(x)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class MultiScalePatchEmbedding(nn.Module):
    """Multi-scale patch embedding for capturing different temporal patterns."""
    
    def __init__(self, seq_length: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        # Different patch sizes for multi-scale features
        self.patch_sizes = [8, 16, 32, 64]  # Different temporal scales
        self.valid_patch_sizes = []
        self.embeddings = nn.ModuleList()
        
        # Only keep patch sizes that divide seq_length evenly
        for patch_size in self.patch_sizes:
            if seq_length % patch_size == 0:
                self.valid_patch_sizes.append(patch_size)
        
        # Calculate embedding dimension per scale
        num_scales = len(self.valid_patch_sizes)
        if num_scales == 0:
            raise ValueError(f"No valid patch sizes found for seq_length={seq_length}")
        
        # Each scale gets the full embedding dimension initially
        for i, patch_size in enumerate(self.valid_patch_sizes):
            self.embeddings.append(
                PatchEmbedding1D(seq_length, patch_size, in_channels, embed_dim)
            )
        
        # Project each scale to the target embedding dimension (no concatenation)
        # Each scale will produce embeddings of size embed_dim
        # We'll combine them through attention rather than concatenation
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embeddings: (batch_size, total_patches, embed_dim)
            patch_info: (batch_size, total_patches) - patch scale information
        """
        all_patches = []
        patch_scales = []
        
        for i, embedding in enumerate(self.embeddings):
            patches = embedding(x)  # (B, num_patches, embed_dim)
            all_patches.append(patches)
            
            # Track which scale each patch belongs to
            num_patches = patches.shape[1]
            scales = torch.full((patches.shape[0], num_patches), i, device=x.device)
            patch_scales.append(scales)
        
        # Concatenate all patches - now they all have the same embed_dim
        embeddings = torch.cat(all_patches, dim=1)  # (B, total_patches, embed_dim)
        patch_scales = torch.cat(patch_scales, dim=1)  # (B, total_patches)
        
        return embeddings, patch_scales


class CrossChannelAttention(nn.Module):
    """Cross-attention mechanism between different EEG channels."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, channel_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_tokens: (batch_size, num_channels, embed_dim)
        """
        # Self-attention across channels
        attn_out, _ = self.cross_attention(channel_tokens, channel_tokens, channel_tokens)
        
        # Residual connection and normalization
        out = self.norm(channel_tokens + self.dropout(attn_out))
        
        return out


class HierarchicalTransformerBlock(nn.Module):
    """Hierarchical transformer block with patch-level and channel-level processing."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Patch-level self-attention
        self.patch_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.patch_norm1 = nn.LayerNorm(embed_dim)
        self.patch_dropout1 = nn.Dropout(dropout)
        
        # MLP for patch processing
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        activation_fn = get_activation_class(activation, class_name=self.__class__.__name__)
        
        self.patch_mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.patch_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_patches, embed_dim)
            mask: Optional attention mask
        """
        # Patch-level self-attention
        attn_out, _ = self.patch_attention(x, x, x, attn_mask=mask)
        x = self.patch_norm1(x + self.patch_dropout1(attn_out))
        
        # MLP
        mlp_out = self.patch_mlp(x)
        x = self.patch_norm2(x + mlp_out)
        
        return x


class AdvancedEEGTransformer(nn.Module):
    """Advanced transformer model for EEG classification with multi-scale processing."""
    
    def __init__(
        self,
        seq_length: int,
        in_channels: int,
        out_dims: int,
        use_age: str = "no",  # Changed default to "no"
        embed_dim: int = 512,
        num_heads: int = 16,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        fc_stages: int = 2,
        **kwargs
    ):
        super().__init__()
        
        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"use_age must be one of ['fc', 'conv', 'no'], got {use_age}")
        
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.use_age = use_age
        self.embed_dim = embed_dim
        self.fc_stages = fc_stages
        
        actual_in_channels = in_channels
        if use_age == "conv":
            actual_in_channels += 1
        
        # Multi-scale patch embedding
        self.patch_embedding = MultiScalePatchEmbedding(seq_length, actual_in_channels, embed_dim)
        
        # Calculate number of patches for each scale
        patch_sizes = [8, 16, 32, 64]
        valid_patch_sizes = [ps for ps in patch_sizes if seq_length % ps == 0]
        self.total_patches = sum(seq_length // ps for ps in valid_patch_sizes)
        
        # Learnable positional encoding for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_patches, embed_dim) * 0.02)
        
        # Scale embedding to distinguish different patch scales
        num_scales = len(valid_patch_sizes)
        self.scale_embedding = nn.Parameter(torch.randn(num_scales, embed_dim) * 0.02)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Channel tokens for channel-specific processing (not used in current implementation)
        self.channel_tokens = nn.Parameter(torch.randn(1, actual_in_channels, embed_dim) * 0.02)
        
        # Cross-channel attention (not used in current implementation)
        self.cross_channel_attention = CrossChannelAttention(embed_dim, num_heads, dropout)
        
        # Main transformer layers
        self.transformer_layers = nn.ModuleList([
            HierarchicalTransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, activation
            ) for _ in range(num_layers)
        ])
        
        # Layer norm before classification head
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        head_input_dim = embed_dim
        if use_age == "fc":
            head_input_dim += 1
            
        activation_fn = get_activation_class(activation, class_name=self.__class__.__name__)
        
        # Build classification head
        head_layers = OrderedDict()
        current_dim = head_input_dim
        
        for i in range(fc_stages - 1):
            head_layers[f"linear_{i+1}"] = nn.Linear(current_dim, current_dim // 2)
            head_layers[f"dropout_{i+1}"] = nn.Dropout(dropout)
            head_layers[f"activation_{i+1}"] = activation_fn()
            current_dim = current_dim // 2
            
        head_layers["output"] = nn.Linear(current_dim, out_dims)
        self.classification_head = nn.Sequential(head_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def get_num_fc_stages(self):
        return self.fc_stages
    
    def compute_feature_embedding(self, x: torch.Tensor, age: torch.Tensor = None, target_from_last: int = 0):
        """
        Compute feature embeddings from input signals.
        
        Args:
            x: (batch_size, channels, seq_length)
            age: (batch_size,) - optional, only used if use_age != "no"
            target_from_last: How many layers from the end to return features from
        """
        B = x.shape[0]
        
        # Handle age as convolutional input
        if self.use_age == "conv" and age is not None:
            age_expanded = age.view(B, 1, 1).expand(B, 1, x.shape[2])
            x = torch.cat([x, age_expanded], dim=1)
        
        # Multi-scale patch embedding
        patch_embeddings, patch_scales = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        
        # Add scale embeddings - now dimensions match
        for i in range(patch_scales.max().item() + 1):
            mask = patch_scales == i
            if mask.any():
                # Scale embedding has the same dimension as patch embeddings
                patch_embeddings[mask] += self.scale_embedding[i]
        
        # Add positional embeddings
        patch_embeddings = patch_embeddings + self.pos_embedding
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Extract class token
        cls_output = x[:, 0]  # (B, embed_dim)
        cls_output = self.norm(cls_output)
        
        # Add age information if needed
        if self.use_age == "fc" and age is not None:
            cls_output = torch.cat([cls_output, age.unsqueeze(1)], dim=1)
        
        # Apply classification head up to target_from_last
        if target_from_last == 0:
            output = self.classification_head(cls_output)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(f"target_from_last ({target_from_last}) > fc_stages ({self.fc_stages})")
            
            layers = list(self.classification_head.children())
            num_to_apply = len(layers) - target_from_last
            
            output = cls_output
            for layer in layers[:num_to_apply]:
                output = layer(output)
        
        return output
    
    def forward(self, x: torch.Tensor, age: torch.Tensor = None) -> torch.Tensor:
        """Forward pass."""
        return self.compute_feature_embedding(x, age)


def create_advanced_eeg_transformer(
    seq_length: int,
    in_channels: int, 
    out_dims: int,
    model_size: str = "base",
    **kwargs
) -> AdvancedEEGTransformer:
    """Create an advanced EEG transformer with predefined configurations."""
    
    configs = {
        "tiny": {
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "mlp_ratio": 4.0
        },
        "small": {
            "embed_dim": 384,
            "num_heads": 12,
            "num_layers": 8,
            "mlp_ratio": 4.0
        },
        "base": {
            "embed_dim": 512,
            "num_heads": 16,
            "num_layers": 12,
            "mlp_ratio": 4.0
        },
        "large": {
            "embed_dim": 768,
            "num_heads": 24,
            "num_layers": 16,
            "mlp_ratio": 4.0
        },
        "huge": {
            "embed_dim": 1024,
            "num_heads": 32,
            "num_layers": 20,
            "mlp_ratio": 4.0
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    return AdvancedEEGTransformer(
        seq_length=seq_length,
        in_channels=in_channels,
        out_dims=out_dims,
        **config
    )
