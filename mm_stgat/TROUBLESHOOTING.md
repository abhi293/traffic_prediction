# NaN Values Troubleshooting Guide

## Problem: NaN Loss and MAE during Training

### Root Causes

1. **Exploding Gradients** - With 2000 nodes, gradients can explode
2. **Numerical Instability in Attention** - Softmax overflow with large scores
3. **Learning Rate Too High** - 0.001 may be too aggressive for large graphs
4. **Data Normalization Issues** - Extreme values in input data

### Fixes Applied

#### 1. Attention Numerical Stability ✅
- **Clamped attention scores** to [-10, 10] range
- **Replaced `-inf` with `-10`** in attention masks
- **Added NaN detection** in softmax output with uniform fallback

#### 2. Input Validation ✅
- Check for NaN/Inf in inputs and replace with zeros
- Early detection in training loop with skip mechanism

#### 3. Gradient Clipping ✅
- Already implemented: `max_norm=1.0`

#### 4. NaN Detection in Training Loop ✅
- Skip batches with NaN outputs
- Print diagnostic information

### Recommended Solutions

#### Option 1: Reduce Learning Rate (EASIEST)
```bash
# For 2000 nodes, use 10x smaller learning rate
python train.py --batch_size 8 --num_epochs 50 --max_nodes 2000 --learning_rate 0.0001
```

#### Option 2: Use Smaller Batch Size
```bash
# Reduce batch size to 4 for more stable training
python train.py --batch_size 4 --num_epochs 50 --max_nodes 2000 --learning_rate 0.0005
```

#### Option 3: Reduce Number of Nodes
```bash
# Use 1000 nodes instead of 2000
python train.py --batch_size 8 --num_epochs 50 --max_nodes 1000
```

#### Option 4: Increase Gradient Clipping
Edit `train.py` line ~256:
```python
# Change from
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
# To
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
```

### Best Configuration for 2000 Nodes

```bash
python train.py \
  --batch_size 4 \
  --num_epochs 50 \
  --max_nodes 2000 \
  --learning_rate 0.0001 \
  --weight_decay 0.0001 \
  --patience 15
```

### Why NaN Happens with Larger Graphs

| Nodes | Edges | Attention Computations | Gradient Magnitude |
|-------|-------|------------------------|-------------------|
| 500   | 5K    | ~250K per layer       | Normal            |
| 1000  | 10K   | ~1M per layer         | 2-4x larger       |
| 2000  | 20K   | ~4M per layer         | 4-16x larger      |

**Key Insight**: Gradient magnitude grows roughly O(n²) with number of nodes in graph attention networks.

### Diagnostic Commands

Check if NaN appears early:
```bash
# Run for just 1 epoch to see when NaN appears
python train.py --batch_size 8 --num_epochs 1 --max_nodes 2000
```

Check data statistics:
```python
import pandas as pd
import numpy as np

df = pd.read_csv('uk_traffic_prediction_ready_2020_2024.csv')
print("Traffic features statistics:")
for col in ['all_motor_vehicles', 'cars_and_taxis', 'LGVs', 'all_HGVs']:
    print(f"{col}:")
    print(f"  Min: {df[col].min()}")
    print(f"  Max: {df[col].max()}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Std: {df[col].std():.2f}")
    print(f"  NaN count: {df[col].isna().sum()}")
```

### Signs of Improvement

✅ **Good Training**:
```
Batch [0/14] - Loss: 0.0035 | MAE: 0.0072
Batch [10/14] - Loss: 0.0031 | MAE: 0.0068
```

❌ **NaN Training**:
```
Batch [0/14] - Loss: nan | MAE: nan
Batch [10/14] - Loss: nan | MAE: nan
```

### Emergency Fix: Use CPU with Lower Precision

If all else fails, train on CPU (slower but more numerically stable):
```bash
python train.py \
  --device cpu \
  --batch_size 4 \
  --num_epochs 30 \
  --max_nodes 1000 \
  --learning_rate 0.0005
```

### Quick Reference

| Configuration | Learning Rate | Batch Size | Expected Time (50 epochs) |
|--------------|---------------|------------|---------------------------|
| 500 nodes   | 0.001         | 8          | ~12 hours                 |
| 1000 nodes  | 0.0005        | 4          | ~24 hours                 |
| 2000 nodes  | 0.0001        | 4          | ~48 hours                 |

### Code Changes Summary

1. **model.py**:
   - Line 78-81: Clamp attention scores
   - Line 83: Use -10 instead of -inf in mask
   - Line 89-92: NaN fallback in attention
   - Line 346-353: Input validation

2. **train.py**:
   - Line 260-265: NaN detection with skip
   - Line 256: Gradient clipping (already present)

These changes make the model more robust to numerical instability while maintaining performance.
