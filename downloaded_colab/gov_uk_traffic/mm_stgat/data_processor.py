"""
Data preprocessing and graph construction for MM-STGAT
Handles data loading, normalization, graph construction, and dataset creation
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist
import pickle
import os


class TrafficDataPreprocessor:
    """
    Preprocesses traffic data for MM-STGAT model
    """
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.node_id_map = {}
        self.reverse_node_map = {}
        
    def load_data(self, file_path):
        """Load and validate traffic data"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        print(f"Loaded {len(df):,} records")
        print(f"Columns: {df.columns.tolist()}")
        
        # Validate critical columns
        required_cols = ['count_point_id', 'year', 'count_date', 'hour', 
                        'all_motor_vehicles', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date to datetime
        df['count_date'] = pd.to_datetime(df['count_date'])
        
        # Sort by location and time
        df = df.sort_values(['count_point_id', 'count_date', 'hour']).reset_index(drop=True)
        
        print(f"Date range: {df['count_date'].min()} to {df['count_date'].max()}")
        print(f"Unique count points: {df['count_point_id'].nunique()}")
        
        return df
    
    def construct_graph(self, df, method='knn', k=5, distance_threshold=None):
        """
        Construct spatial graph from traffic count points
        
        Args:
            df: DataFrame with location data
            method: 'knn' (k-nearest neighbors) or 'distance' (threshold-based)
            k: Number of nearest neighbors for KNN
            distance_threshold: Distance threshold in km for distance-based method
        """
        print(f"\nConstructing spatial graph using {method} method...")
        
        # Get unique nodes with their locations
        nodes = df.groupby('count_point_id').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        # Create node ID mapping
        self.node_id_map = {node_id: idx for idx, node_id in enumerate(nodes['count_point_id'])}
        self.reverse_node_map = {idx: node_id for node_id, idx in self.node_id_map.items()}
        
        num_nodes = len(nodes)
        print(f"Number of nodes: {num_nodes}")
        
        # Extract coordinates
        coords = nodes[['latitude', 'longitude']].values
        
        # Compute pairwise distances (Haversine approximation in km)
        distances = self._haversine_distance_matrix(coords)
        
        # Construct adjacency matrix
        if method == 'knn':
            adj_matrix = self._knn_adjacency(distances, k)
        elif method == 'distance':
            if distance_threshold is None:
                distance_threshold = np.percentile(distances[distances > 0], 10)
            adj_matrix = self._distance_adjacency(distances, distance_threshold)
        else:
            raise ValueError(f"Unknown graph construction method: {method}")
        
        # Add self-loops
        np.fill_diagonal(adj_matrix, 1)
        
        print(f"Adjacency matrix shape: {adj_matrix.shape}")
        print(f"Number of edges: {(adj_matrix > 0).sum() - num_nodes}")
        print(f"Average degree: {(adj_matrix.sum(axis=1) - 1).mean():.2f}")
        
        return adj_matrix, nodes
    
    def _haversine_distance_matrix(self, coords):
        """Compute haversine distance matrix (in km)"""
        lat, lon = coords[:, 0], coords[:, 1]
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        lat1 = lat_rad[:, np.newaxis]
        lat2 = lat_rad[np.newaxis, :]
        lon1 = lon_rad[:, np.newaxis]
        lon2 = lon_rad[np.newaxis, :]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        distance = 6371 * c
        
        return distance
    
    def _knn_adjacency(self, distances, k):
        """Construct KNN adjacency matrix"""
        num_nodes = distances.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            # Get k nearest neighbors (excluding self)
            nearest = np.argsort(distances[i])[1:k+1]
            adj_matrix[i, nearest] = 1
            adj_matrix[nearest, i] = 1  # Make it symmetric
        
        return adj_matrix
    
    def _distance_adjacency(self, distances, threshold):
        """Construct distance-based adjacency matrix"""
        adj_matrix = (distances <= threshold).astype(float)
        return adj_matrix
    
    def prepare_features(self, df):
        """
        Prepare input features and targets
        """
        print("\nPreparing features...")
        
        # Traffic volume features (main targets and inputs)
        traffic_features = ['all_motor_vehicles', 'cars_and_taxis', 'LGVs', 'all_HGVs']
        traffic_features = [f for f in traffic_features if f in df.columns]
        
        # Temporal features
        temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend']
        temporal_features = [f for f in temporal_features if f in df.columns]
        
        # Categorical features
        categorical_features = ['road_category', 'road_type', 'direction_of_travel']
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Encode categorical features
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(
                    df[col].fillna('Unknown').astype(str)
                )
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(
                    df[col].fillna('Unknown').astype(str)
                )
        
        # Continuous auxiliary features
        auxiliary_features = ['latitude', 'longitude', 'link_length_km'] + \
                            [f + '_encoded' for f in categorical_features]
        auxiliary_features = [f for f in auxiliary_features if f in df.columns]
        
        # Fill NaN values in auxiliary features
        for col in auxiliary_features:
            if col in df.columns:
                if df[col].isna().any():
                    # Fill with median for numerical, 0 for encoded
                    if 'encoded' in col:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(df[col].median())
                # Replace inf values
                df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        print(f"Traffic features: {traffic_features}")
        print(f"Temporal features: {temporal_features}")
        print(f"Auxiliary features: {auxiliary_features}")
        
        return traffic_features, temporal_features, auxiliary_features
    
    def create_sequences(self, df, traffic_features, temporal_features, auxiliary_features):
        """
        Create sequences for time-series prediction
        """
        print("\nCreating sequences...")
        
        seq_len = self.config['seq_len']
        pred_len = self.config['pred_len']
        
        X_list = []  # Input sequences
        Y_list = []  # Target sequences
        aux_list = []  # Auxiliary features
        timestamps = []
        
        # Group by count_point_id
        grouped = df.groupby('count_point_id')
        
        for count_point_id, group in grouped:
            if count_point_id not in self.node_id_map:
                continue
            
            node_idx = self.node_id_map[count_point_id]
            
            # Sort by datetime
            group = group.sort_values(['count_date', 'hour']).reset_index(drop=True)
            
            # Extract features
            traffic_data = group[traffic_features].values
            temporal_data = group[temporal_features].values
            aux_data = group[auxiliary_features].iloc[0].values  # Static for each node
            
            # Fill NaN/Inf in auxiliary data
            aux_data = np.nan_to_num(aux_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create sliding windows
            for i in range(len(group) - seq_len - pred_len + 1):
                # Input sequence
                x_traffic = traffic_data[i:i+seq_len]
                x_temporal = temporal_data[i:i+seq_len]
                x = np.concatenate([x_traffic, x_temporal], axis=1)
                
                # Target sequence
                y = traffic_data[i+seq_len:i+seq_len+pred_len]
                
                X_list.append((node_idx, x))
                Y_list.append((node_idx, y))
                aux_list.append((node_idx, aux_data))
                timestamps.append(group.iloc[i+seq_len]['count_date'])
        
        print(f"Created {len(X_list)} sequences")
        
        return X_list, Y_list, aux_list, timestamps
    
    def normalize_data(self, X_list, Y_list, aux_list, fit=True):
        """
        Normalize features
        """
        print("\nNormalizing data...")
        
        num_nodes = len(self.node_id_map)
        
        # Initialize arrays
        X_normalized = {}
        Y_normalized = {}
        aux_normalized = {}
        
        # Collect all data for fitting scaler
        if fit:
            all_X = np.concatenate([x for _, x in X_list], axis=0)
            all_Y = np.concatenate([y for _, y in Y_list], axis=0)
            all_aux = np.array([aux for _, aux in aux_list])
            
            self.scaler_X = StandardScaler()
            self.scaler_Y = StandardScaler()
            self.scaler_aux = StandardScaler()
            
            self.scaler_X.fit(all_X)
            self.scaler_Y.fit(all_Y)
            self.scaler_aux.fit(all_aux)
        
        # Normalize sequences
        for (node_idx, x), (_, y), (_, aux) in zip(X_list, Y_list, aux_list):
            x_norm = self.scaler_X.transform(x)
            y_norm = self.scaler_Y.transform(y)
            aux_norm = self.scaler_aux.transform(aux.reshape(1, -1)).flatten()
            
            if node_idx not in X_normalized:
                X_normalized[node_idx] = []
                Y_normalized[node_idx] = []
                aux_normalized[node_idx] = aux_norm
            
            X_normalized[node_idx].append(x_norm)
            Y_normalized[node_idx].append(y_norm)
        
        return X_normalized, Y_normalized, aux_normalized
    
    def save_preprocessor(self, path):
        """Save preprocessor state"""
        state = {
            'scaler_X': self.scaler_X,
            'scaler_Y': self.scaler_Y,
            'scaler_aux': self.scaler_aux,
            'node_id_map': self.node_id_map,
            'reverse_node_map': self.reverse_node_map,
            'label_encoders': self.label_encoders,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"\nSaved preprocessor to {path}")
    
    def load_preprocessor(self, path):
        """Load preprocessor state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.scaler_X = state['scaler_X']
        self.scaler_Y = state['scaler_Y']
        self.scaler_aux = state['scaler_aux']
        self.node_id_map = state['node_id_map']
        self.reverse_node_map = state['reverse_node_map']
        self.label_encoders = state['label_encoders']
        self.config = state['config']
        print(f"Loaded preprocessor from {path}")


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for traffic data
    """
    def __init__(self, X_data, Y_data, aux_data, num_nodes):
        """
        Args:
            X_data: Dict {node_idx: list of sequences}
            Y_data: Dict {node_idx: list of targets}
            aux_data: Dict {node_idx: auxiliary features}
            num_nodes: Total number of nodes
        """
        self.num_nodes = num_nodes
        self.samples = []
        
        # Find maximum number of sequences across all nodes
        max_sequences = max(len(seqs) for seqs in X_data.values())
        
        # Create aligned samples
        for seq_idx in range(max_sequences):
            X_sample = []
            Y_sample = []
            
            for node_idx in range(num_nodes):
                if node_idx in X_data and seq_idx < len(X_data[node_idx]):
                    X_sample.append(X_data[node_idx][seq_idx])
                    Y_sample.append(Y_data[node_idx][seq_idx])
                else:
                    # Pad with zeros if node doesn't have enough sequences
                    if node_idx in X_data:
                        X_sample.append(np.zeros_like(X_data[node_idx][0]))
                        Y_sample.append(np.zeros_like(Y_data[node_idx][0]))
                    else:
                        # Use dummy data
                        X_sample.append(np.zeros((X_data[list(X_data.keys())[0]][0].shape)))
                        Y_sample.append(np.zeros((Y_data[list(Y_data.keys())[0]][0].shape)))
            
            # Stack into tensors
            X_tensor = torch.FloatTensor(np.array(X_sample))  # [num_nodes, seq_len, features]
            Y_tensor = torch.FloatTensor(np.array(Y_sample))  # [num_nodes, pred_len, features]
            
            # Auxiliary features - handle missing nodes gracefully
            aux_list = []
            for i in range(num_nodes):
                if i in aux_data:
                    aux_list.append(aux_data[i])
                else:
                    # Use zeros for missing nodes
                    if len(aux_data) > 0:
                        aux_list.append(np.zeros_like(list(aux_data.values())[0]))
                    else:
                        aux_list.append(np.zeros(1))  # Dummy value
            aux_tensor = torch.FloatTensor(np.array(aux_list))
            
            self.samples.append((X_tensor, Y_tensor, aux_tensor))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_dataloaders(X_train, Y_train, aux_train, X_val, Y_val, aux_val,
                       X_test, Y_test, aux_test, num_nodes, batch_size):
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    """
    train_dataset = TrafficDataset(X_train, Y_train, aux_train, num_nodes)
    val_dataset = TrafficDataset(X_val, Y_val, aux_val, num_nodes)
    test_dataset = TrafficDataset(X_test, Y_test, aux_test, num_nodes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    print(f"\nDataLoader created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
