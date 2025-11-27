"""
Training script for MM-STGAT on a subset of nodes
This script trains the model on a sampled subset of traffic monitoring points
to handle memory constraints with the full 14,071 node graph.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import MM_STGAT, MM_STGAT_Loss
from data_processor import TrafficDataPreprocessor
from visualize import ResultsVisualizer
from torch.utils.data import Dataset


class SimpleTrafficDataset(Dataset):
    """Simple dataset for traffic prediction"""
    def __init__(self, X, Y, aux):
        self.X = X
        self.Y = Y
        self.aux = aux
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'y': self.Y[idx],
            'aux': self.aux[idx]
        }


def setup_device():
    """Setup computation device with fallback: CUDA -> DirectML -> CPU"""
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device('cuda')
    
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("Using DirectML (AMD GPU)")
        return dml_device
    except ImportError:
        print("DirectML not available")
    
    print("Using CPU")
    return torch.device('cpu')


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class SubsetTrainer:
    """Training pipeline for MM-STGAT on node subset"""
    def __init__(self, config, device, subset_size=1000):
        self.config = config
        self.device = device
        self.subset_size = subset_size
        
        # Load and preprocess data with node subset
        print(f"\nLoading data and sampling {subset_size} nodes...")
        self.load_data_subset()
        
        # Update config with actual dimensions
        self.config['num_nodes'] = self.num_nodes
        self.config['input_dim'] = self.input_dim
        self.config['output_dim'] = self.output_dim
        self.config['auxiliary_dim'] = self.auxiliary_dim
        
        print(f"\nInitializing model with {self.num_nodes} nodes...")
        self.model = MM_STGAT(self.config).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Loss function
        self.criterion = MM_STGAT_Loss(
            alpha=config.get('loss_alpha', 0.7),
            beta=config.get('loss_beta', 0.2),
            gamma=config.get('loss_gamma', 0.1)
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.0001)
        )
        
        # TensorBoard
        log_dir = f"logs/mm_stgat_subset{subset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        
    def load_data_subset(self):
        """Load data and sample subset of nodes"""
        # Load full dataset
        df = pd.read_csv('../uk_traffic_prediction_ready_2020_2024.csv')
        
        # Get unique nodes and sample subset
        all_nodes = df['count_point_id'].unique()
        np.random.seed(self.config.get('seed', 42))
        selected_nodes = np.random.choice(all_nodes, size=min(self.subset_size, len(all_nodes)), replace=False)
        
        print(f"Selected {len(selected_nodes)} out of {len(all_nodes)} nodes")
        
        # Filter dataset to selected nodes
        df_subset = df[df['count_point_id'].isin(selected_nodes)].copy()
        
        print(f"Subset data: {len(df_subset)} records")
        
        # Create preprocessor and process data
        self.preprocessor = TrafficDataPreprocessor(self.config)
        
        # Construct graph
        print("Constructing spatial graph...")
        self.adj_matrix = self.preprocessor.construct_graph(
            df_subset,
            method=self.config.get('graph_method', 'knn'),
            k=self.config.get('graph_k', 8)
        )
        
        # Prepare features
        print("Preparing features...")
        traffic_features, temporal_features, auxiliary_features = self.preprocessor.prepare_features(df_subset)
        
        # Create sequences
        print("Creating sequences...")
        X_list, Y_list, aux_list, timestamps = self.preprocessor.create_sequences(
            df_subset, 
            traffic_features, 
            temporal_features, 
            auxiliary_features
        )
        
        # Split into train/val/test
        total_samples = len(X_list)
        train_size = int(self.config.get('train_ratio', 0.7) * total_samples)
        val_size = int(self.config.get('val_ratio', 0.15) * total_samples)
        
        X_train = X_list[:train_size]
        Y_train = Y_list[:train_size]
        aux_train = aux_list[:train_size]
        
        X_val = X_list[train_size:train_size+val_size]
        Y_val = Y_list[train_size:train_size+val_size]
        aux_val = aux_list[train_size:train_size+val_size]
        
        X_test = X_list[train_size+val_size:]
        Y_test = Y_list[train_size+val_size:]
        aux_test = aux_list[train_size+val_size:]
        
        # Reorganize and normalize data
        print("Organizing and normalizing data...")
        
        # Group sequences by timestamp index (same for all nodes)
        # Find the minimum number of sequences across all nodes
        min_sequences = min(len([x for node_idx, x in X_list if node_idx == nid]) for nid in range(self.num_nodes))
        
        # Organize data: [num_samples, num_nodes, seq_len, features]
        X_organized_train = []
        Y_organized_train = []
        aux_organized_train = []
        
        X_organized_val = []
        Y_organized_val = []
        aux_organized_val = []
        
        X_organized_test = []
        Y_organized_test = []
        aux_organized_test = []
        
        # Group by nodes for each split
        for split_name, X_split, Y_split, aux_split, X_out, Y_out, aux_out in [
            ('train', X_train, Y_train, aux_train, X_organized_train, Y_organized_train, aux_organized_train),
            ('val', X_val, Y_val, aux_val, X_organized_val, Y_organized_val, aux_organized_val),
            ('test', X_test, Y_test, aux_test, X_organized_test, Y_organized_test, aux_organized_test)
        ]:
            # Group by node
            node_data_X = {i: [] for i in range(self.num_nodes)}
            node_data_Y = {i: [] for i in range(self.num_nodes)}
            node_data_aux = {}
            
            for (node_idx, x), (_, y), (_, aux) in zip(X_split, Y_split, aux_split):
                node_data_X[node_idx].append(x)
                node_data_Y[node_idx].append(y)
                node_data_aux[node_idx] = aux
            
            # Find max sequences for this split
            max_seq = max(len(seqs) for seqs in node_data_X.values() if len(seqs) > 0)
            
            # Create samples: each sample has data from all nodes at same time index
            for seq_idx in range(max_seq):
                X_sample = []
                Y_sample = []
                
                for node_idx in range(self.num_nodes):
                    if seq_idx < len(node_data_X[node_idx]):
                        X_sample.append(node_data_X[node_idx][seq_idx])
                        Y_sample.append(node_data_Y[node_idx][seq_idx])
                    else:
                        # Pad if needed
                        X_sample.append(np.zeros_like(node_data_X[node_idx][0]) if len(node_data_X[node_idx]) > 0 else np.zeros((self.config['seq_len'], 8)))
                        Y_sample.append(np.zeros_like(node_data_Y[node_idx][0]) if len(node_data_Y[node_idx]) > 0 else np.zeros((self.config['pred_len'], 4)))
                
                X_out.append(np.array(X_sample))
                Y_out.append(np.array(Y_sample))
                aux_out.append(np.array([node_data_aux[i] for i in range(self.num_nodes)]))
        
        # Convert to numpy arrays
        X_train_array = np.array(X_organized_train)  # [samples, nodes, seq_len, features]
        Y_train_array = np.array(Y_organized_train)  # [samples, nodes, pred_len, features]
        aux_train_array = np.array(aux_organized_train)  # [samples, nodes, aux_features]
        
        X_val_array = np.array(X_organized_val)
        Y_val_array = np.array(Y_organized_val)
        aux_val_array = np.array(aux_organized_val)
        
        X_test_array = np.array(X_organized_test)
        Y_test_array = np.array(Y_organized_test)
        aux_test_array = np.array(aux_organized_test)
        
        # Normalize
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        scaler_aux = StandardScaler()
        
        # Fit scalers on training data
        scaler_X.fit(X_train_array.reshape(-1, X_train_array.shape[-1]))
        scaler_Y.fit(Y_train_array.reshape(-1, Y_train_array.shape[-1]))
        scaler_aux.fit(aux_train_array.reshape(-1, aux_train_array.shape[-1]))
        
        # Transform
        X_train_norm = scaler_X.transform(X_train_array.reshape(-1, X_train_array.shape[-1])).reshape(X_train_array.shape)
        Y_train_norm = scaler_Y.transform(Y_train_array.reshape(-1, Y_train_array.shape[-1])).reshape(Y_train_array.shape)
        aux_train_norm = scaler_aux.transform(aux_train_array.reshape(-1, aux_train_array.shape[-1])).reshape(aux_train_array.shape)
        
        X_val_norm = scaler_X.transform(X_val_array.reshape(-1, X_val_array.shape[-1])).reshape(X_val_array.shape)
        Y_val_norm = scaler_Y.transform(Y_val_array.reshape(-1, Y_val_array.shape[-1])).reshape(Y_val_array.shape)
        aux_val_norm = scaler_aux.transform(aux_val_array.reshape(-1, aux_val_array.shape[-1])).reshape(aux_val_array.shape)
        
        X_test_norm = scaler_X.transform(X_test_array.reshape(-1, X_test_array.shape[-1])).reshape(X_test_array.shape)
        Y_test_norm = scaler_Y.transform(Y_test_array.reshape(-1, Y_test_array.shape[-1])).reshape(Y_test_array.shape)
        aux_test_norm = scaler_aux.transform(aux_test_array.reshape(-1, aux_test_array.shape[-1])).reshape(aux_test_array.shape)
        
        # Get dimensions
        self.num_nodes = len(selected_nodes)
        self.input_dim = X_train_norm.shape[3]  # features per timestep
        self.output_dim = Y_train_norm.shape[3]
        self.auxiliary_dim = aux_train_norm.shape[2]
        
        print(f"Data shapes - X_train: {X_train_norm.shape}, Y_train: {Y_train_norm.shape}, aux_train: {aux_train_norm.shape}")
        print(f"Graph shape: {self.adj_matrix.shape}, edges: {(self.adj_matrix > 0).sum()}")
        print(f"Dimensions - Nodes: {self.num_nodes}, Input: {self.input_dim}, Output: {self.output_dim}, Aux: {self.auxiliary_dim}")
        
        # Convert to tensors and transpose to [batch, seq_len, nodes, features]
        X_train_tensor = torch.FloatTensor(X_train_norm).transpose(1, 2)  # [batch, seq_len, nodes, features]
        Y_train_tensor = torch.FloatTensor(Y_train_norm).transpose(1, 2)  # [batch, pred_len, nodes, features]
        aux_train_tensor = torch.FloatTensor(aux_train_norm)  # [batch, nodes, aux_features]
        
        X_val_tensor = torch.FloatTensor(X_val_norm).transpose(1, 2)
        Y_val_tensor = torch.FloatTensor(Y_val_norm).transpose(1, 2)
        aux_val_tensor = torch.FloatTensor(aux_val_norm)
        
        X_test_tensor = torch.FloatTensor(X_test_norm).transpose(1, 2)
        Y_test_tensor = torch.FloatTensor(Y_test_norm).transpose(1, 2)
        aux_test_tensor = torch.FloatTensor(aux_test_norm)
        
        # Create datasets
        train_dataset = SimpleTrafficDataset(X_train_tensor, Y_train_tensor, aux_train_tensor)
        val_dataset = SimpleTrafficDataset(X_val_tensor, Y_val_tensor, aux_val_tensor)
        test_dataset = SimpleTrafficDataset(X_test_tensor, Y_test_tensor, aux_test_tensor)
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 32)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        adj_matrix = self.adj_matrix.to(self.device)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            X = batch['x'].to(self.device)
            Y = batch['y'].to(self.device)
            aux = batch['aux'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            Y_pred, _, _ = self.model(X, adj_matrix, aux)
            
            # Calculate loss
            loss, loss_dict = self.criterion(Y_pred, Y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mae += loss_dict['mae']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mae': f"{loss_dict['mae']:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self, data_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        adj_matrix = self.adj_matrix.to(self.device)
        
        with torch.no_grad():
            for batch in data_loader:
                X = batch['x'].to(self.device)
                Y = batch['y'].to(self.device)
                aux = batch['aux'].to(self.device)
                
                Y_pred, _, _ = self.model(X, adj_matrix, aux)
                loss, loss_dict = self.criterion(Y_pred, Y)
                
                total_loss += loss.item()
                total_mae += loss_dict['mae']
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def test(self):
        """Test model and collect predictions"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        adj_matrix = self.adj_matrix.to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                X = batch['x'].to(self.device)
                Y = batch['y'].to(self.device)
                aux = batch['aux'].to(self.device)
                
                Y_pred, _, _ = self.model(X, adj_matrix, aux)
                
                all_predictions.append(Y_pred.cpu())
                all_targets.append(Y.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        loss, loss_dict = self.criterion(predictions, targets)
        
        print(f"\nTest Results:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  MAE: {loss_dict['mae']:.4f}")
        print(f"  MSE: {loss_dict['mse']:.4f}")
        
        return predictions, targets, loss_dict
    
    def train(self, num_epochs):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('='*60)
            
            # Train
            train_loss, train_mae = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mae = self.validate(self.val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('MAE', {
                'train': train_mae,
                'val': val_mae
            }, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            print(f"\nTrain Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, 'checkpoints/best_model_subset.pt')
                print("✓ Saved best model")
            
            # Check early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        
        # Load best model for testing
        print("\nLoading best model for testing...")
        checkpoint = torch.load('checkpoints/best_model_subset.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test
        predictions, targets, test_metrics = self.test()
        self.history['test_loss'] = [test_metrics['mse']]
        self.history['test_mae'] = [test_metrics['mae']]
        
        return predictions, targets


def main():
    """Main training function"""
    print("="*60)
    print("MM-STGAT Training on Node Subset")
    print("="*60)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Create trainer with subset of nodes
    subset_size = 1000  # Train on 1000 nodes instead of 14,071
    trainer = SubsetTrainer(config, device, subset_size=subset_size)
    
    # Train
    num_epochs = config.get('num_epochs', 100)
    predictions, targets = trainer.train(num_epochs)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs('results/plots', exist_ok=True)
    
    visualizer = ResultsVisualizer()
    
    # Denormalize predictions for visualization
    pred_np = predictions.numpy()
    target_np = targets.numpy()
    
    # Reshape for visualization
    pred_2d = pred_np.reshape(-1, pred_np.shape[-1])
    target_2d = target_np.reshape(-1, target_np.shape[-1])
    
    # Generate plots
    visualizer.plot_training_curves(trainer.history, 'results/plots/training_curves.png')
    visualizer.plot_predictions_vs_actual(pred_2d, target_2d, 'results/plots/predictions_vs_actual.png')
    visualizer.plot_error_distribution(pred_2d, target_2d, 'results/plots/error_distribution.png')
    visualizer.plot_scatter_predictions(pred_2d, target_2d, 'results/plots/scatter_predictions.png')
    visualizer.plot_metrics_summary(trainer.history, 'results/plots/metrics_summary.png')
    
    # Generate report
    visualizer.generate_report(trainer.history, pred_2d, target_2d, 'results/training_report.txt')
    
    print("\n✓ Training complete! Results saved in 'results/' directory")
    print(f"✓ Model trained on {subset_size} nodes (subset of 14,071 total nodes)")
    print("✓ To train on full dataset, increase GPU memory or use a compute cluster")


if __name__ == '__main__':
    main()
