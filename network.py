import torch.nn as nn
from torch.nn.functional import normalize
import torch

import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),  # 改
            nn.ReLU(),
            nn.Linear(1024, feature_dim),  # 改
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),  # 改
            nn.ReLU(),
            nn.Linear(1024, 512),  # 改
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class CrossViewAttention(nn.Module):
    """
    A Cross-View Attention module that allows latent representations from different
    views to interact and fuse information. This is implemented as a standard
    Transformer Encoder block.
    """
    def __init__(self, feature_dim, num_heads=4):
        super(CrossViewAttention, self).__init__()
        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        # Layer normalization
        self.layer_norm2 = nn.LayerNorm(feature_dim)

    def forward(self, zs):
        # zs is a list of tensors, each of shape (batch_size, feature_dim)
        # Stack the list of tensors into a single tensor for attention
        # Shape becomes (num_views, batch_size, feature_dim)
        z_stack = torch.stack(zs, dim=0)

        # Apply self-attention across the views.
        # Q=z_stack, K=z_stack, V=z_stack
        attn_output, _ = self.attention(z_stack, z_stack, z_stack)

        # Add & Norm (Residual connection)
        z_stack = self.layer_norm1(z_stack + attn_output)

        # Feed Forward Network
        ffn_output = self.ffn(z_stack)

        # Add & Norm (Second residual connection)
        z_stack = self.layer_norm2(z_stack + ffn_output)

        # Unstack the tensor back into a list of tensors
        fused_zs = [z_stack[v] for v in range(z_stack.shape[0])]
        return fused_zs

# --- MODIFIED: Main Network with Attention ---
class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim, num_heads=2):
        super(Network, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.views = views

        for v in range(views):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))

        # Instantiate the cross-view attention module
        self.cross_view_attention = CrossViewAttention(feature_dim, num_heads)

    def forward(self, xs):
        # 1. Get initial latent representation for each view independently
        initial_zs = []
        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            initial_zs.append(z)

        # 2. Apply the cross-view attention to fuse information across views
        fused_zs = self.cross_view_attention(initial_zs)

        # 3. Decode each view using its corresponding FUSED representation
        xrs = []
        for v in range(self.views):
            xr = self.decoders[v](fused_zs[v])
            xrs.append(xr)

        # Return the fused latent representations and the reconstructions
        return fused_zs, xrs




