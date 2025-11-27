"""
Multimodal Spatial-Temporal Graph Attention Network (MM-STGAT) for Traffic Prediction

This implementation includes:
- Multi-head spatial attention mechanism
- Temporal attention with LSTM/GRU
- Multimodal feature fusion
- Robust architecture for traffic forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """
    Spatial Graph Attention Layer
    Computes attention coefficients between nodes in the traffic network
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2):
        super(SpatialAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention parameters
        self.head_dim = out_features // num_heads
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        
        # Output projection
        self.W_o = nn.Linear(out_features, out_features)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, x, adj_matrix):
        """
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.size()
        
        # Linear projections and split into multiple heads
        Q = self.W_q(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Q, K, V shape: [batch_size, num_heads, num_nodes, head_dim]
        
        # For large graphs, compute attention in very small chunks to avoid memory issues
        chunk_size = 256  # Process only 256 nodes at a time
        attention_outputs = []
        
        for i in range(0, num_nodes, chunk_size):
            end_i = min(i + chunk_size, num_nodes)
            Q_chunk = Q[:, :, i:end_i, :].contiguous()  # [batch, heads, chunk_nodes, head_dim]
            
            # For each query chunk, compute attention with all keys in sub-chunks
            chunk_context_parts = []
            
            for j in range(0, num_nodes, chunk_size):
                end_j = min(j + chunk_size, num_nodes)
                K_chunk = K[:, :, j:end_j, :].contiguous()
                V_chunk = V[:, :, j:end_j, :].contiguous()
                
                # Compute attention scores for this sub-chunk
                scores_chunk = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                # Clamp scores to prevent overflow/underflow
                scores_chunk = torch.clamp(scores_chunk, min=-10, max=10)
                
                # Apply adjacency mask if provided
                if adj_matrix is not None:
                    mask_chunk = adj_matrix[i:end_i, j:end_j].unsqueeze(0).unsqueeze(0)
                    scores_chunk = scores_chunk.masked_fill(mask_chunk == 0, -10)  # Use -10 instead of -inf
                
                # Softmax over this chunk
                attn_chunk = F.softmax(scores_chunk, dim=-1)
                # Replace NaN values with uniform attention if they occur
                if torch.isnan(attn_chunk).any():
                    attn_chunk = torch.where(torch.isnan(attn_chunk), 
                                            torch.ones_like(attn_chunk) / attn_chunk.size(-1),
                                            attn_chunk)
                attn_chunk = self.dropout_layer(attn_chunk)
                
                # Apply attention to values for this chunk
                context_part = torch.matmul(attn_chunk, V_chunk)
                chunk_context_parts.append(context_part)
                
                # Clear intermediate tensors to save memory
                del scores_chunk, attn_chunk, K_chunk
            
            # Sum the context from all value chunks (approximate full attention)
            context_chunk = torch.stack(chunk_context_parts, dim=0).sum(dim=0)
            attention_outputs.append(context_chunk)
            
            # Clear intermediate tensors
            del Q_chunk, chunk_context_parts, context_chunk
        
        # Concatenate all query chunks
        context = torch.cat(attention_outputs, dim=2)  # [batch, heads, num_nodes, head_dim]
        attn_weights_final = None  # Skip returning attention weights to save memory
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.out_features)
        output = self.W_o(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + x if x.size(-1) == self.out_features else output)
        
        return output, attn_weights_final


class TemporalAttention(nn.Module):
    """
    Temporal Attention Layer with 1D Convolution
    Captures temporal dependencies without RNNs for DirectML compatibility
    Uses 1D conv + multi-head attention (fully DirectML-compatible)
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 1D Convolution for temporal encoding (DirectML-compatible)
        # Two conv layers to capture temporal patterns
        self.temporal_conv1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        
        # Attention mechanism
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_k = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_v = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: Temporal features [batch_size, seq_len, hidden_dim]
        Returns:
            Attended temporal features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 1D Convolution encoding: need [batch, channels, seq_len]
        x_conv = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        conv_out = self.temporal_conv1(x_conv)
        conv_out = self.activation(conv_out)
        conv_out = self.temporal_conv2(conv_out)
        conv_out = self.activation(conv_out)
        conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Multi-head attention
        Q = self.W_q(conv_out).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(conv_out).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(conv_out).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_dim)
        
        output = self.W_o(context)
        output = self.layer_norm(output + x)
        
        return output, attn_weights


class MultimodalFusion(nn.Module):
    """
    Multimodal Feature Fusion Layer
    Combines spatial, temporal, and auxiliary features
    """
    def __init__(self, spatial_dim, temporal_dim, auxiliary_dim, output_dim, dropout=0.2):
        super(MultimodalFusion, self).__init__()
        
        total_dim = spatial_dim + temporal_dim + auxiliary_dim
        
        # Fusion network
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Attention-based gating mechanism
        self.gate_spatial = nn.Linear(spatial_dim, 1)
        self.gate_temporal = nn.Linear(temporal_dim, 1)
        self.gate_auxiliary = nn.Linear(auxiliary_dim, 1)
        
    def forward(self, spatial_features, temporal_features, auxiliary_features):
        """
        Args:
            spatial_features: [batch_size, spatial_dim]
            temporal_features: [batch_size, temporal_dim]
            auxiliary_features: [batch_size, auxiliary_dim]
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Gating mechanism
        gate_s = torch.sigmoid(self.gate_spatial(spatial_features))
        gate_t = torch.sigmoid(self.gate_temporal(temporal_features))
        gate_a = torch.sigmoid(self.gate_auxiliary(auxiliary_features))
        
        # Normalize gates
        gate_sum = gate_s + gate_t + gate_a
        gate_s = gate_s / gate_sum
        gate_t = gate_t / gate_sum
        gate_a = gate_a / gate_sum
        
        # Apply gating
        spatial_gated = spatial_features * gate_s
        temporal_gated = temporal_features * gate_t
        auxiliary_gated = auxiliary_features * gate_a
        
        # Concatenate and fuse
        combined = torch.cat([spatial_gated, temporal_gated, auxiliary_gated], dim=-1)
        fused = self.fusion_layers(combined)
        
        return fused


class MM_STGAT(nn.Module):
    """
    Multimodal Spatial-Temporal Graph Attention Network
    
    Architecture:
    1. Input embedding
    2. Spatial attention layers (graph structure)
    3. Temporal attention layers (time series)
    4. Multimodal fusion
    5. Output prediction layers
    """
    def __init__(self, config):
        super(MM_STGAT, self).__init__()
        
        self.num_nodes = config['num_nodes']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_spatial_layers = config['num_spatial_layers']
        self.num_temporal_layers = config['num_temporal_layers']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.auxiliary_dim = config.get('auxiliary_dim', 10)
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Spatial attention layers
        self.spatial_layers = nn.ModuleList([
            SpatialAttention(
                self.hidden_dim,
                self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(self.num_spatial_layers)
        ])
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(
                self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(self.num_temporal_layers)
        ])
        
        # Auxiliary feature processing
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(self.auxiliary_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Multimodal fusion
        self.fusion = MultimodalFusion(
            spatial_dim=self.hidden_dim,
            temporal_dim=self.hidden_dim,
            auxiliary_dim=self.hidden_dim // 2,
            output_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # Output projection
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.pred_len * self.output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, adj_matrix, auxiliary_features):
        """
        Args:
            x: Input features [batch_size, seq_len, num_nodes, input_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            auxiliary_features: Auxiliary features [batch_size, num_nodes, auxiliary_dim]
        Returns:
            predictions: [batch_size, num_nodes, pred_len, output_dim]
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        
        # Input validation - check for NaN or Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ Warning: Input x contains NaN or Inf values")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        if torch.isnan(auxiliary_features).any() or torch.isinf(auxiliary_features).any():
            print(f"⚠️ Warning: Auxiliary features contain NaN or Inf values")
            auxiliary_features = torch.where(torch.isnan(auxiliary_features) | torch.isinf(auxiliary_features), 
                                            torch.zeros_like(auxiliary_features), auxiliary_features)
        
        # Input embedding
        x = x.reshape(batch_size * seq_len, num_nodes, self.input_dim)
        x = self.input_embedding(x)  # [batch_size * seq_len, num_nodes, hidden_dim]
        
        # Spatial attention
        spatial_attn_weights = []
        for spatial_layer in self.spatial_layers:
            x, attn_w = spatial_layer(x, adj_matrix)
            spatial_attn_weights.append(attn_w)
        
        spatial_features = x.reshape(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        # Temporal attention (process nodes in batches to save memory)
        temporal_features = []
        temporal_attn_weights = []
        node_batch_size = 256  # Process 256 nodes at a time
        
        for start_idx in range(0, num_nodes, node_batch_size):
            end_idx = min(start_idx + node_batch_size, num_nodes)
            batch_nodes = spatial_features[:, :, start_idx:end_idx, :]  # [batch_size, seq_len, batch_nodes, hidden_dim]
            
            # Process each node in this mini-batch
            for node_idx in range(batch_nodes.size(2)):
                node_seq = batch_nodes[:, :, node_idx, :]  # [batch_size, seq_len, hidden_dim]
                for temporal_layer in self.temporal_layers:
                    node_seq, attn_w = temporal_layer(node_seq)
                temporal_features.append(node_seq[:, -1, :])  # Take last timestep
                if len(temporal_attn_weights) < num_nodes:
                    temporal_attn_weights.append(attn_w)
        
        temporal_features = torch.stack(temporal_features, dim=1)  # [batch_size, num_nodes, hidden_dim]
        
        # Process auxiliary features
        auxiliary_encoded = self.auxiliary_encoder(auxiliary_features)  # [batch_size, num_nodes, hidden_dim // 2]
        
        # Multimodal fusion (process nodes in batches to save memory)
        fused_features = []
        node_batch_size = 512  # Process 512 nodes at a time
        
        for start_idx in range(0, num_nodes, node_batch_size):
            end_idx = min(start_idx + node_batch_size, num_nodes)
            
            for node_idx in range(start_idx, end_idx):
                spatial_feat = spatial_features[:, -1, node_idx, :]  # Last timestep spatial features
                temporal_feat = temporal_features[:, node_idx, :]
                auxiliary_feat = auxiliary_encoded[:, node_idx, :]
                
                fused = self.fusion(spatial_feat, temporal_feat, auxiliary_feat)
                fused_features.append(fused)
        
        fused_features = torch.stack(fused_features, dim=1)  # [batch_size, num_nodes, hidden_dim]
        
        # Output prediction
        output = self.output_layers(fused_features)  # [batch_size, num_nodes, pred_len * output_dim]
        output = output.reshape(batch_size, num_nodes, self.pred_len, self.output_dim)
        
        # Transpose to match target shape [batch, pred_len, num_nodes, output_dim]
        output = output.permute(0, 2, 1, 3)  # [batch_size, pred_len, num_nodes, output_dim]
        
        return output, spatial_attn_weights, temporal_attn_weights


class MM_STGAT_Loss(nn.Module):
    """
    Custom loss function for MM-STGAT
    Combines MSE, MAE, and smoothness regularization
    """
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(MM_STGAT_Loss, self).__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # MAE weight
        self.gamma = gamma  # Smoothness weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, num_nodes, pred_len, output_dim]
            targets: [batch_size, num_nodes, pred_len, output_dim]
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # MAE loss
        mae_loss = F.l1_loss(predictions, targets)
        
        # Temporal smoothness regularization
        temporal_diff = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        smoothness_loss = torch.mean(temporal_diff ** 2)
        
        # Combined loss
        total_loss = self.alpha * mse_loss + self.beta * mae_loss + self.gamma * smoothness_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'smoothness': smoothness_loss.item(),
            'total': total_loss.item()
        }
