"""
Training script for MM-STGAT with GPU support (CUDA -> DirectML -> CPU fallback)
Includes robust training pipeline with early stopping, checkpointing, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from datetime import datetime
import sys
import argparse

from model import MM_STGAT, MM_STGAT_Loss
from data_processor import TrafficDataPreprocessor, prepare_dataloaders
from visualize import ResultsVisualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MM-STGAT model for traffic prediction')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset CSV file')
    parser.add_argument('--seq_len', type=int, default=None, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=None, help='Prediction length')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension size')
    parser.add_argument('--num_spatial_layers', type=int, default=None, help='Number of spatial attention layers')
    parser.add_argument('--num_temporal_layers', type=int, default=None, help='Number of temporal attention layers')
    parser.add_argument('--num_heads', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience')
    
    # Graph parameters
    parser.add_argument('--graph_method', type=str, default=None, choices=['knn', 'distance'], help='Graph construction method')
    parser.add_argument('--graph_k', type=int, default=None, help='K for KNN graph')
    
    # Other
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'dml', 'cpu'], help='Force specific device (cuda/dml/cpu)')
    parser.add_argument('--no_visualization', action='store_true', help='Skip automatic visualization generation')
    parser.add_argument('--max_nodes', type=int, default=None, help='Limit number of nodes for testing (randomly sample)')
    
    return parser.parse_args()


def setup_device(force_device=None):
    """
    Setup device with fallback: CUDA -> DirectML -> CPU
    
    Args:
        force_device: Force specific device ('cuda', 'dml', 'cpu', or None for auto)
    """
    print("=" * 80)
    print("DEVICE SETUP")
    print("=" * 80)
    
    # Force specific device if requested
    if force_device:
        if force_device == 'cpu':
            device = torch.device('cpu')
            print(f"✓ Using CPU (forced)")
            print(f"  Cores: {os.cpu_count()}")
            return device
        elif force_device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✓ Using CUDA (forced)")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                return device
            else:
                print(f"✗ CUDA not available, falling back to CPU")
                device = torch.device('cpu')
                print(f"✓ Using CPU")
                print(f"  Cores: {os.cpu_count()}")
                return device
        elif force_device == 'dml':
            try:
                import torch_directml
                device = torch_directml.device()
                print(f"✓ Using DirectML (forced)")
                return device
            except (ImportError, RuntimeError) as e:
                print(f"✗ DirectML not available: {e}")
                print(f"  Falling back to CPU")
                device = torch.device('cpu')
                print(f"✓ Using CPU")
                print(f"  Cores: {os.cpu_count()}")
                return device
    
    # Auto-detect with fallback
    # Try CUDA first
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    
    # Try DirectML (for AMD GPUs on Windows)
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"✓ Using DirectML (AMD GPU acceleration)")
        return device
    except (ImportError, RuntimeError) as e:
        print(f"  DirectML not available: {e}")
    
    # Fallback to CPU
    device = torch.device('cpu')
    print(f"✓ Using CPU")
    print(f"  Cores: {os.cpu_count()}")
    
    return device


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Training manager for MM-STGAT
    """
    def __init__(self, model, config, device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Loss function
        self.criterion = MM_STGAT_Loss(
            alpha=config.get('loss_alpha', 0.7),
            beta=config.get('loss_beta', 0.2),
            gamma=config.get('loss_gamma', 0.1)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.0001)
        )
        
        # TensorBoard
        log_dir = f"logs/mm_stgat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_mae': [],
            'val_mae': [],
            'test_mae': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
    def train_epoch(self, train_loader, adj_matrix, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        adj_matrix = adj_matrix.to(self.device)
        
        # Enable mixed precision training for memory efficiency
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for batch_idx, (X, Y, aux) in enumerate(train_loader):
            # Move to device
            X = X.to(self.device)  # [batch_size, num_nodes, seq_len, features]
            Y = Y.to(self.device)  # [batch_size, num_nodes, pred_len, features]
            aux = aux.to(self.device)  # [batch_size, num_nodes, aux_features]
            
            # Reshape X to [batch_size, seq_len, num_nodes, features]
            X = X.permute(0, 2, 1, 3)
            Y = Y.permute(0, 2, 1, 3)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, spatial_attn, temporal_attn = self.model(X, adj_matrix, aux)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, Y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Check for NaN in outputs
            if torch.isnan(loss) or torch.isnan(predictions).any():
                print(f"\n⚠️ Warning: NaN detected at batch {num_batches}")
                print(f"  Loss: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                print(f"  Predictions NaN count: {torch.isnan(predictions).sum().item()}")
                # Skip this batch
                continue
            
            # Metrics
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - Y)).item()
            
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
            # Log batch progress
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] - "
                      f"Loss: {loss.item():.4f} | MAE: {mae:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self, val_loader, adj_matrix):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        adj_matrix = adj_matrix.to(self.device)
        
        with torch.no_grad():
            for X, Y, aux in val_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                aux = aux.to(self.device)
                
                X = X.permute(0, 2, 1, 3)
                Y = Y.permute(0, 2, 1, 3)
                
                predictions, _, _ = self.model(X, adj_matrix, aux)
                
                loss, loss_dict = self.criterion(predictions, Y)
                mae = torch.mean(torch.abs(predictions - Y)).item()
                
                total_loss += loss.item()
                total_mae += mae
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def test(self, test_loader, adj_matrix, preprocessor=None):
        """Test the model and return predictions"""
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        total_smape = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        adj_matrix = adj_matrix.to(self.device)
        
        with torch.no_grad():
            for X, Y, aux in test_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                aux = aux.to(self.device)
                
                X = X.permute(0, 2, 1, 3)
                Y = Y.permute(0, 2, 1, 3)
                
                predictions, spatial_attn, temporal_attn = self.model(X, adj_matrix, aux)
                
                loss, loss_dict = self.criterion(predictions, Y)
                
                # Compute metrics on normalized data
                mae = torch.mean(torch.abs(predictions - Y))
                rmse = torch.sqrt(torch.mean((predictions - Y) ** 2))
                # sMAPE (Symmetric MAPE) - more robust than MAPE
                smape = torch.mean(2 * torch.abs(predictions - Y) / (torch.abs(predictions) + torch.abs(Y) + 1e-8)) * 100
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_rmse += rmse.item()
                total_smape += smape.item()
                num_batches += 1
                
                # Store predictions
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = total_rmse / num_batches
        avg_smape = total_smape / num_batches
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics in original scale if preprocessor provided
        if preprocessor is not None:
            # Reshape for inverse transform: [batch, pred_len, nodes, features] -> [batch*pred_len*nodes, features]
            pred_shape = all_predictions.shape
            pred_flat = all_predictions.reshape(-1, pred_shape[-1])
            target_flat = all_targets.reshape(-1, pred_shape[-1])
            
            # Inverse transform (only traffic features, first 'output_dim' features)
            pred_original = preprocessor.scaler_Y.inverse_transform(pred_flat)
            target_original = preprocessor.scaler_Y.inverse_transform(target_flat)
            
            # Compute metrics on original scale
            mae_original = np.mean(np.abs(pred_original - target_original))
            rmse_original = np.sqrt(np.mean((pred_original - target_original) ** 2))
            # sMAPE on original scale
            smape_original = np.mean(2 * np.abs(pred_original - target_original) / 
                                    (np.abs(pred_original) + np.abs(target_original) + 1e-8)) * 100
            
            metrics_original = {
                'mae_original': mae_original,
                'rmse_original': rmse_original,
                'smape_original': smape_original
            }
        else:
            metrics_original = {}
        
        metrics = {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': avg_rmse,
            'smape': avg_smape,
            **metrics_original
        }
        
        return metrics, all_predictions, all_targets, spatial_attn, temporal_attn
    
    def train(self, train_loader, val_loader, adj_matrix, num_epochs):
        """Full training loop"""
        print("\n" + "=" * 80)
        print("TRAINING START")
        print("=" * 80)
        
        adj_matrix_tensor = torch.FloatTensor(adj_matrix)
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 80)
            
            # Training
            train_loss, train_mae = self.train_epoch(train_loader, adj_matrix_tensor, epoch)
            
            # Validation
            val_loss, val_mae = self.validate(val_loader, adj_matrix_tensor)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('MAE/train', train_mae, epoch)
            self.writer.add_scalar('MAE/val', val_mae, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\n  Results:")
            print(f"    Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}")
            print(f"    Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.4f}")
            print(f"    LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                break
        
        self.writer.close()
        
        # Save final history
        self.save_history()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        # Convert device to string for DirectML compatibility
        device_str = str(self.device)
        if 'privateuseone' in device_str:
            device_str = 'cpu'  # Load to CPU first for DirectML
        
        checkpoint = torch.load(checkpoint_path, map_location=device_str)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch'] + 1
        
        # Move model to correct device after loading
        self.model = self.model.to(self.device)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    
    def save_history(self):
        """Save training history"""
        history_path = os.path.join('results', 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"\nTraining history saved to {history_path}")


def main(args):
    """Main training function"""
    
    # Load configuration from file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command-line arguments (if provided)
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and key != 'config' and key != 'no_visualization' and key != 'device':
            config[key] = value
    
    print("=" * 80)
    print("MM-STGAT TRAFFIC PREDICTION")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup device
    device = setup_device(force_device=args.device)
    
    # Data preprocessing
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    preprocessor = TrafficDataPreprocessor(config)
    
    # Load data
    df = preprocessor.load_data(config['data_path'])
    
    # Sample nodes if max_nodes is specified
    if args.max_nodes and args.max_nodes > 0:
        unique_nodes = df['count_point_id'].unique()
        if len(unique_nodes) > args.max_nodes:
            np.random.seed(config.get('seed', 42))
            sampled_nodes = np.random.choice(unique_nodes, size=args.max_nodes, replace=False)
            df = df[df['count_point_id'].isin(sampled_nodes)].copy()
            print(f"\n⚠ Sampled {args.max_nodes} nodes out of {len(unique_nodes)} for testing")
    
    # Construct graph
    adj_matrix, nodes = preprocessor.construct_graph(
        df, 
        method=config['graph_method'],
        k=config['graph_k']
    )
    
    config['num_nodes'] = len(nodes)
    
    # Prepare features
    traffic_features, temporal_features, auxiliary_features = preprocessor.prepare_features(df)
    
    config['input_dim'] = len(traffic_features) + len(temporal_features)
    config['output_dim'] = len(traffic_features)
    config['auxiliary_dim'] = len(auxiliary_features)
    
    # Create sequences
    X_list, Y_list, aux_list, timestamps = preprocessor.create_sequences(
        df, traffic_features, temporal_features, auxiliary_features
    )
    
    print(f"Created {len(X_list)} sequences")
    
    # Count sequences per node for better reporting
    sequences_per_node = {}
    for node_idx, _ in X_list:
        sequences_per_node[node_idx] = sequences_per_node.get(node_idx, 0) + 1
    
    avg_sequences = np.mean(list(sequences_per_node.values()))
    print(f"Average sequences per node: {avg_sequences:.1f}")
    print(f"Min sequences per node: {min(sequences_per_node.values())}")
    print(f"Max sequences per node: {max(sequences_per_node.values())}")
    
    # Split data (70% train, 15% val, 15% test)
    total_samples = len(X_list)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    X_train = {node: [] for node in range(config['num_nodes'])}
    Y_train = {node: [] for node in range(config['num_nodes'])}
    X_val = {node: [] for node in range(config['num_nodes'])}
    Y_val = {node: [] for node in range(config['num_nodes'])}
    X_test = {node: [] for node in range(config['num_nodes'])}
    Y_test = {node: [] for node in range(config['num_nodes'])}
    
    for i in range(total_samples):
        node_idx, x = X_list[i]
        _, y = Y_list[i]
        
        if i < train_size:
            X_train[node_idx].append(x)
            Y_train[node_idx].append(y)
        elif i < train_size + val_size:
            X_val[node_idx].append(x)
            Y_val[node_idx].append(y)
        else:
            X_test[node_idx].append(x)
            Y_test[node_idx].append(y)
    
    # Get auxiliary features - create a proper mapping for all nodes
    aux_dict = {}
    for node_idx, aux in aux_list:
        if node_idx not in aux_dict:
            aux_dict[node_idx] = aux
    
    # Ensure all nodes have auxiliary features
    aux_train = {}
    aux_val = {}
    aux_test = {}
    for node_idx in range(config['num_nodes']):
        if node_idx in aux_dict:
            aux_train[node_idx] = aux_dict[node_idx]
            aux_val[node_idx] = aux_dict[node_idx]
            aux_test[node_idx] = aux_dict[node_idx]
        else:
            # Use zero vector for missing nodes
            aux_train[node_idx] = np.zeros(config['auxiliary_dim'])
            aux_val[node_idx] = np.zeros(config['auxiliary_dim'])
            aux_test[node_idx] = np.zeros(config['auxiliary_dim'])
    
    # Normalize
    X_train_norm, Y_train_norm, aux_train_norm = preprocessor.normalize_data(
        [(n, x) for n in X_train for x in X_train[n]],
        [(n, y) for n in Y_train for y in Y_train[n]],
        [(n, aux_train[n]) for n in range(config['num_nodes'])],
        fit=True
    )
    
    X_val_norm, Y_val_norm, aux_val_norm = preprocessor.normalize_data(
        [(n, x) for n in X_val for x in X_val[n]],
        [(n, y) for n in Y_val for y in Y_val[n]],
        [(n, aux_val[n]) for n in range(config['num_nodes'])],
        fit=False
    )
    
    X_test_norm, Y_test_norm, aux_test_norm = preprocessor.normalize_data(
        [(n, x) for n in X_test for x in X_test[n]],
        [(n, y) for n in Y_test for y in Y_test[n]],
        [(n, aux_test[n]) for n in range(config['num_nodes'])],
        fit=False
    )
    
    # Create directories if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Save preprocessor
    preprocessor.save_preprocessor('checkpoints/preprocessor.pkl')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train_norm, Y_train_norm, aux_train_norm,
        X_val_norm, Y_val_norm, aux_val_norm,
        X_test_norm, Y_test_norm, aux_test_norm,
        config['num_nodes'],
        config['batch_size']
    )
    
    # Initialize model
    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)
    
    model = MM_STGAT(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: MM-STGAT")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Trainer(model, config, device)
    
    # Train
    trainer.train(train_loader, val_loader, adj_matrix, config['num_epochs'])
    
    # Test
    print("\n" + "=" * 80)
    print("TESTING")
    print("=" * 80)
    
    # Load best model
    trainer.load_checkpoint('checkpoints/best_model.pth')
    
    test_metrics, predictions, targets, spatial_attn, temporal_attn = trainer.test(
        test_loader, torch.FloatTensor(adj_matrix), preprocessor
    )
    
    print(f"\nTest Results (Normalized Scale):")
    print(f"  Loss:  {test_metrics['loss']:.4f}")
    print(f"  MAE:   {test_metrics['mae']:.4f}")
    print(f"  RMSE:  {test_metrics['rmse']:.4f}")
    print(f"  sMAPE: {test_metrics['smape']:.2f}%")
    
    if 'mae_original' in test_metrics:
        print(f"\nTest Results (Original Scale):")
        print(f"  MAE:   {test_metrics['mae_original']:.2f} vehicles/hour")
        print(f"  RMSE:  {test_metrics['rmse_original']:.2f} vehicles/hour")
        print(f"  sMAPE: {test_metrics['smape_original']:.2f}%")
    
    # Save test results
    np.savez('results/test_predictions.npz',
             predictions=predictions,
             targets=targets,
             metrics=test_metrics)
    
    # Generate visualizations (unless disabled)
    if not args.no_visualization:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Automatically generate all visualizations
        try:
            visualizer = ResultsVisualizer()
            visualizer.generate_all_plots()
            print("\n✓ All visualizations generated successfully!")
        except Exception as e:
            print(f"\n⚠ Warning: Visualization generation failed: {e}")
            print("You can manually run: python visualize.py")
    else:
        print("\n⚠ Visualization generation skipped (--no_visualization flag)")
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED!")
    print("=" * 80)
    print("\nFiles saved:")
    print("  - checkpoints/best_model.pth")
    print("  - checkpoints/preprocessor.pkl")
    print("  - results/training_history.json")
    print("  - results/test_predictions.npz")
    print("  - results/plots/*.png (all visualizations)")
    print("  - results/performance_report.txt")


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    main(args)
