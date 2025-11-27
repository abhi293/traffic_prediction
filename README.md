# MM-STGAT: Multimodal Spatial-Temporal Graph Attention Network for Traffic Prediction

A robust and efficient deep learning model for traffic volume prediction using Multimodal Spatial-Temporal Graph Attention Networks (MM-STGAT). This implementation includes GPU acceleration support (CUDA → DirectML → CPU fallback), comprehensive visualization tools, and production-ready training pipeline.

## 🌟 Features

- **Advanced Architecture**: Multi-head spatial and temporal attention mechanisms with multimodal feature fusion
- **GPU Acceleration**: Automatic device detection with fallback support
  - ✅ NVIDIA CUDA (primary)
  - ✅ AMD DirectML (Windows) - **Fully optimized for DirectML compatibility**
  - ✅ CPU (fallback)
- **DirectML Optimized**: Uses 1D convolutions instead of RNNs for better AMD GPU performance
- **Memory Efficient**: Chunked attention mechanism for large-scale graphs (14,000+ nodes)
- **Flexible Training**: Comprehensive command-line arguments for all hyperparameters
- **Subset Training**: `--max_nodes` argument to test with smaller subsets before full-scale training
- **Robust Training**: Early stopping, learning rate scheduling, gradient clipping, and checkpointing
- **Comprehensive Visualization**: Training curves, prediction analysis, error distribution, and performance metrics
- **Graph Construction**: KNN and distance-based spatial graph methods
- **Production Ready**: Modular code structure, configuration management, and extensive logging

## 📋 Table of Contents

- [DirectML Compatibility](#directml-compatibility)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Visualization](#visualization)
- [Model Architecture](#model-architecture)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

---

## 💻 DirectML Compatibility

### Important Notes for AMD GPU Users (DirectML)

This MM-STGAT implementation is **fully compatible with AMD GPUs via DirectML**, with the following optimizations:

#### Architecture Modifications
- **Temporal Layer**: Uses **1D Convolutions** instead of LSTM/GRU
  - RNNs (LSTM/GRU) fall back to CPU on DirectML, causing severe performance degradation
  - 1D Conv layers are fully DirectML-accelerated and maintain similar modeling capability
  
- **Spatial Attention**: Implements **chunked attention** mechanism
  - Processes attention in 256-node chunks to avoid memory overflow
  - Required for large graphs (14,000+ nodes) on DirectML
  - Approximates full attention while staying within GPU memory limits

#### Performance Considerations
- **DirectML Speed**: Due to chunked processing, DirectML training is slower than CUDA
- **Recommended Approaches**:
  1. **For testing**: Use `--max_nodes 500` to train on subset: `python train.py --max_nodes 500`
  2. **For production**: Use CPU instead: `python train.py --device cpu --batch_size 8`
  3. **Best performance**: Use NVIDIA GPU with CUDA if available

#### Why These Changes?
DirectML has stricter memory limits and different operator support compared to CUDA:
- RNN operations (`aten::_thnn_fused_lstm_cell`, `aten::_thnn_fused_gru_cell`) fall back to CPU
- Large attention matrices (14,071 x 14,071) cause out-of-memory errors
- Chunked processing trades speed for memory efficiency

**The model is still MM-STGAT** with all its multimodal spatial-temporal capabilities intact!

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- GPU (optional but recommended):
  - NVIDIA GPU with CUDA support, OR
  - AMD GPU with DirectML support (Windows)
- 8GB+ RAM (16GB+ recommended)

### Step 1: Clone or Navigate to Project Directory

```bash
cd mm_stgat
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**For CUDA Support (NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For DirectML Support (AMD GPU on Windows):**
```bash
pip install torch-directml
```

### Step 4: Verify Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## 📊 Dataset Preparation

### Required Dataset

The model expects a preprocessed traffic dataset with the following features:

**Required Columns:**
- `count_point_id`: Unique identifier for each traffic monitoring point
- `year`: Year of observation
- `count_date`: Date of measurement (YYYY-MM-DD)
- `hour`: Hour of day (0-23)
- `latitude`, `longitude`: Spatial coordinates
- `all_motor_vehicles`: Total traffic volume (target variable)
- Additional vehicle type columns (optional): `cars_and_taxis`, `LGVs`, `all_HGVs`, etc.
- Temporal features: `day_of_week`, `month`, `is_weekend`
- Road features: `road_category`, `road_type`, `direction_of_travel`

### Dataset Location

Place your preprocessed dataset in the parent directory:
```
d:\research\Traffic\gov_uk_traffic\uk_traffic_prediction_ready_2020_2024.csv
```

Or update the `data_path` in `config.json`.

### Data Preprocessing

If you haven't preprocessed your data yet, use the preprocessing script:
```bash
cd ..
python preprocess_for_prediction.py
```

This will:
- Filter data by year range (2020-2024)
- Extract temporal features
- Handle missing values appropriately
- Create a clean dataset for training

---

## ⚙️ Configuration

Edit `config.json` to customize model parameters:

### Data Configuration
```json
{
  "data_path": "../uk_traffic_prediction_ready_2020_2024.csv",
  "seq_len": 12,        // Input sequence length (hours)
  "pred_len": 6         // Prediction horizon (hours)
}
```

### Model Architecture
```json
{
  "hidden_dim": 128,              // Hidden layer dimension
  "num_spatial_layers": 3,        // Number of spatial attention layers
  "num_temporal_layers": 2,       // Number of temporal attention layers
  "num_heads": 4,                 // Number of attention heads
  "dropout": 0.2                  // Dropout rate
}
```

### Graph Construction
```json
{
  "graph_method": "knn",          // "knn" or "distance"
  "graph_k": 8,                   // Number of nearest neighbors (for KNN)
  "graph_distance_threshold": 50  // Distance threshold in km (for distance method)
}
```

### Training Parameters
```json
{
  "batch_size": 32,
  "num_epochs": 100,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "patience": 15,                 // Early stopping patience
  "min_delta": 0.0001            // Minimum improvement threshold
}
```

### Loss Function Weights
```json
{
  "loss_alpha": 0.7,    // MSE weight
  "loss_beta": 0.2,     // MAE weight
  "loss_gamma": 0.1     // Smoothness regularization weight
}
```

---

## 🎓 Training

### Start Training

```bash
# Basic training with default config
python train.py

# Override specific parameters via command line
python train.py --batch_size 4 --learning_rate 0.0005 --num_epochs 50

# Train with custom batch size for memory constraints
python train.py --batch_size 1 --hidden_dim 32

# Full example with multiple parameters
python train.py \
    --batch_size 2 \
    --hidden_dim 64 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --dropout 0.3 \
    --patience 20

# Skip visualization generation (faster)
python train.py --no_visualization

# Test with subset of nodes (recommended for DirectML)
python train.py --max_nodes 500 --batch_size 2 --num_epochs 10

# Force CPU training (faster than DirectML for large datasets)
python train.py --device cpu --batch_size 8 --num_epochs 50

# Progressive testing: start small, then scale up
python train.py --max_nodes 500 --num_epochs 5   # Quick test
python train.py --max_nodes 2000 --num_epochs 20  # Medium scale
python train.py --batch_size 4 --num_epochs 100   # Full dataset
```

### Command-Line Arguments

You can override any config parameter via command line:

**Data Parameters:**
- `--data_path`: Path to dataset CSV file
- `--seq_len`: Input sequence length (default: 6)
- `--pred_len`: Prediction length (default: 3)
- `--max_nodes`: Limit number of nodes for testing (e.g., `--max_nodes 500` to train on subset)

**Model Parameters:**
- `--hidden_dim`: Hidden dimension size (default: 32)
- `--num_spatial_layers`: Number of spatial attention layers (default: 2)
- `--num_temporal_layers`: Number of temporal attention layers (default: 2)
- `--num_heads`: Number of attention heads (default: 2)
- `--dropout`: Dropout rate (default: 0.2)

**Training Parameters:**
- `--batch_size`: Batch size (default: 1, increase for faster training if GPU allows)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for L2 regularization (default: 0.0001)
- `--patience`: Early stopping patience (default: 15)

**Graph Parameters:**
- `--graph_method`: Graph construction method - `knn` or `distance` (default: knn)
- `--graph_k`: K for KNN graph (default: 8)
- `--graph_distance_threshold`: Distance threshold in km for distance-based graphs (default: 50)

**Other:**
- `--config`: Path to config file (default: config.json)
- `--seed`: Random seed (default: 42)
- `--device`: Force specific device - `cuda`, `dml`, or `cpu` (auto-detect by default)
- `--no_visualization`: Skip automatic visualization generation

### Memory Considerations

**Important**: The full UK traffic dataset contains **14,071 monitoring points**.

**For Full Dataset (14,071 nodes):**
- **DirectML (AMD GPU)**: May be slow with chunked attention. Consider using `--device cpu` for faster training
- **CUDA (NVIDIA GPU with 8GB+)**: Should work with `--batch_size 4 --hidden_dim 64`
- **CPU**: Use `--batch_size 8 --hidden_dim 128` for reasonable speed

**For Subset Testing (recommended to start):**
- Test with smaller subsets first: `--max_nodes 500` or `--max_nodes 1000`
- Gradually increase to find optimal size for your GPU memory
- **Low Memory GPU (<4GB)**: `--max_nodes 500 --batch_size 1 --hidden_dim 32`
- **Medium Memory GPU (4-8GB)**: `--max_nodes 2000 --batch_size 2 --hidden_dim 64`
- **High Memory GPU (>8GB)**: `--max_nodes 5000 --batch_size 4 --hidden_dim 128`

### Training Process

The training script will:

1. **Device Setup**: Automatically detect and configure GPU/CPU (CUDA → DirectML → CPU)
2. **Data Loading**: Load and validate the dataset
3. **Node Sampling** (if using train_subset.py): Select random subset of monitoring points
4. **Graph Construction**: Build spatial graph from traffic monitoring points (KNN or distance-based)
5. **Feature Preparation**: Extract traffic, temporal, and auxiliary features
6. **Data Splitting**: 70% train, 15% validation, 15% test
7. **Normalization**: Standardize features using StandardScaler
8. **Model Training**: Train with early stopping, learning rate scheduling, and checkpointing
9. **Testing**: Evaluate on test set
10. **Visualization**: Automatically generate plots and save results

### Training Output

During training, you'll see:

```
================================================================================
DEVICE SETUP
================================================================================
✓ Using CUDA
  GPU: NVIDIA GeForce RTX 3080
  Memory: 10.00 GB

================================================================================
DATA PREPROCESSING
================================================================================
Loading data from ../uk_traffic_prediction_ready_2020_2024.csv...
Loaded 523,847 records
...

================================================================================
TRAINING START
================================================================================

Epoch [1/100]
--------------------------------------------------------------------------------
  Batch [0/245] - Loss: 0.3421 | MAE: 0.2156
  Batch [10/245] - Loss: 0.2987 | MAE: 0.1923
  ...

  Results:
    Train Loss: 0.2845 | Train MAE: 0.1876
    Val Loss:   0.2901 | Val MAE:   0.1912
    LR: 0.001000 | Time: 45.23s
  ✓ Best model saved (val_loss: 0.2901)
```

### Saved Files

After training completes:

```
mm_stgat/
├── checkpoints/
│   ├── best_model.pth              # Best model weights
│   ├── checkpoint_epoch_10.pth     # Periodic checkpoints
│   └── preprocessor.pkl            # Data preprocessor state
├── results/
│   ├── training_history.json       # Training metrics
│   └── test_predictions.npz        # Test set predictions
└── logs/
    └── mm_stgat_YYYYMMDD_HHMMSS/  # TensorBoard logs
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=logs
```

Then open `http://localhost:6006` in your browser.

---

## 📈 Visualization

### Generate All Visualizations

After training completes, generate comprehensive visualizations:

```bash
python visualize.py
```

### Generated Plots

The visualization script creates:

#### 1. **Training Curves** (`training_curves.png`)
   - Training and validation loss
   - Mean Absolute Error (MAE) progression
   - Learning rate schedule
   - Overfitting gap analysis

#### 2. **Predictions vs Actual** (`predictions_vs_actual.png`)
   - Side-by-side comparison of predicted and actual traffic volumes
   - Multiple samples across different nodes
   - Time-series visualization

#### 3. **Error Analysis** (`error_analysis.png`)
   - Error distribution histogram
   - MAE per sample
   - Error by prediction horizon
   - Q-Q plot for normality testing

#### 4. **Scatter Plots** (`scatter_predictions.png`)
   - Predictions vs actual scatter plot
   - Density hexbin plot
   - Perfect prediction reference line

#### 5. **Metrics Summary** (`metrics_summary.png`)
   - Bar chart of test metrics (Loss, MAE, RMSE, MAPE)
   - Clear value labels

#### 6. **Performance Report** (`performance_report.txt`)
   - Comprehensive text summary
   - Training statistics
   - Test performance metrics
   - Error statistics

### Plot Characteristics

All plots feature:
- ✅ **Bold and enlarged labels**
- ✅ **Clear axis titles**
- ✅ **High-resolution output (300 DPI)**
- ✅ **Professional styling**
- ✅ **Grid lines for readability**

---

## 🏗️ Model Architecture

### MM-STGAT Overview

The Multimodal Spatial-Temporal Graph Attention Network consists of:

```
Input Sequence
     ↓
[Input Embedding]
     ↓
[Spatial Attention Layers] ←── Graph Structure
     ↓
[Temporal Attention Layers] ←── LSTM Encoding
     ↓
[Multimodal Fusion] ←── Auxiliary Features
     ↓
[Output Projection]
     ↓
Predictions
```

### Key Components

#### 1. **Spatial Attention**
- Multi-head graph attention mechanism
- Learns relationships between traffic monitoring points
- Incorporates adjacency matrix for spatial structure
- Captures traffic flow patterns across the network

#### 2. **Temporal Attention**
- **1D Convolutional encoding** for temporal feature extraction (DirectML-optimized)
- Multi-head attention over time steps
- Captures traffic dynamics and trends
- Learns short-term and long-term dependencies
- **Note**: Uses convolutions instead of LSTM/GRU for better DirectML compatibility

#### 3. **Multimodal Fusion**
- Combines spatial, temporal, and auxiliary features
- Gated fusion mechanism
- Learns optimal feature weighting
- Integrates road characteristics, time features, etc.

#### 4. **Loss Function**
- **MSE Loss**: Penalizes large errors
- **MAE Loss**: Robust to outliers
- **Smoothness Regularization**: Ensures realistic temporal transitions

### Model Parameters

Total trainable parameters: **~2-5 million** (depending on configuration)

---

## 📁 File Structure

```
mm_stgat/
├── model.py                    # MM-STGAT architecture
│   ├── SpatialAttention        # Spatial graph attention layer (chunked for memory efficiency)
│   ├── TemporalAttention       # Temporal attention with 1D Conv (DirectML-optimized)
│   ├── MultimodalFusion        # Feature fusion module
│   ├── MM_STGAT                # Main model class
│   └── MM_STGAT_Loss           # Custom loss function
│
├── data_processor.py           # Data preprocessing and graph construction
│   ├── TrafficDataPreprocessor # Data loading and preparation
│   └── TrafficDataset          # PyTorch dataset
│
├── train.py                    # Training script
│   ├── setup_device()          # GPU/CPU detection
│   ├── EarlyStopping           # Early stopping callback
│   └── Trainer                 # Training manager
│
├── visualize.py                # Visualization and analysis
│   └── ResultsVisualizer       # Comprehensive plotting
│
├── config.json                 # Model configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**

**Solution**: Reduce batch size in `config.json`:
```json
{
  "batch_size": 16  // or even 8
}
```

#### 2. **DirectML Not Found (AMD GPU)**

**Solution**: Install DirectML support:
```bash
pip install torch-directml
```

#### 3. **Module Not Found Errors**

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

#### 4. **Data Loading Errors**

**Solution**: Verify dataset path in `config.json` and check that the CSV file exists with required columns.

#### 5. **Low Accuracy / High Loss**

**Solutions**:
- Increase `num_epochs`
- Tune `learning_rate` (try 0.0005 or 0.0001)
- Increase `hidden_dim` (e.g., 256)
- Adjust `graph_k` for better spatial connectivity
- Check data quality and preprocessing

#### 6. **Training Too Slow**

**Solutions**:
- **DirectML users**: Consider using `--device cpu` for faster training on large datasets
- **For testing**: Use `--max_nodes 500` to train on a subset first
- Use GPU if available (CUDA is faster than DirectML for this model)
- Reduce `seq_len` or `pred_len`
- Decrease `num_spatial_layers` or `num_temporal_layers`
- Increase `batch_size` if memory allows

#### 7. **DirectML Slow Performance**

**Solutions**:
- The chunked attention mechanism required for DirectML memory limits is computationally expensive
- **Recommended**: Use CPU for full dataset training: `python train.py --device cpu --batch_size 8`
- Alternatively, train on subset: `python train.py --max_nodes 1000 --batch_size 2`
- Or use NVIDIA GPU with CUDA for best performance

---

## 📊 Expected Performance

### Typical Metrics (After 50-100 Epochs)

- **MAE**: 15-30 vehicles/hour
- **RMSE**: 25-50 vehicles/hour
- **MAPE**: 10-25%

*Performance varies based on dataset quality, graph structure, and hyperparameters.*

---

## 🎯 Usage Tips

### Best Practices

1. **Start with default configuration** - Fine-tune after initial training
2. **Monitor TensorBoard** - Track training progress in real-time
3. **Use early stopping** - Prevent overfitting
4. **Experiment with graph construction** - Try both KNN and distance methods
5. **Check visualizations** - Validate model behavior on test set

### Hyperparameter Tuning Guide

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `hidden_dim` | Model capacity | Start 128, increase if underfitting |
| `num_spatial_layers` | Spatial complexity | 2-4 layers |
| `num_temporal_layers` | Temporal depth | 1-3 layers |
| `num_heads` | Attention granularity | 4-8 heads |
| `dropout` | Regularization | 0.1-0.3 |
| `learning_rate` | Training speed | 0.0001-0.001 |
| `graph_k` | Spatial connectivity | 5-15 neighbors |

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{mm_stgat_traffic,
  title={MM-STGAT: Multimodal Spatial-Temporal Graph Attention Network for Traffic Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mm-stgat-traffic}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- UK Department for Transport for the traffic dataset
- PyTorch team for the deep learning framework
- Research community for spatial-temporal modeling techniques

---

**Last Updated**: November 2024

**Status**: ✅ Production Ready
