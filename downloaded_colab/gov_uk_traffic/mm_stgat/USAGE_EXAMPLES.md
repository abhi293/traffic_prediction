# MM-STGAT Training Examples

## Basic Usage

### 1. Train with Default Settings
```bash
python train.py
```
Uses all settings from `config.json`.

---

## Memory-Constrained Training

### 2. Low Memory GPU (2-4GB VRAM)
```bash
python train.py --batch_size 1 --hidden_dim 32 --num_heads 2
```

### 3. Medium Memory GPU (4-8GB VRAM)
```bash
python train.py --batch_size 4 --hidden_dim 64 --num_heads 4
```

### 4. High Memory GPU (8GB+ VRAM)
```bash
python train.py --batch_size 16 --hidden_dim 128 --num_heads 8
```

---

## Quick Testing

### 5. Fast Test Run (2 epochs)
```bash
python train.py --num_epochs 2 --no_visualization
```

### 6. Quick Validation (smaller model, fewer epochs)
```bash
python train.py --batch_size 1 --hidden_dim 16 --num_epochs 5 --patience 3
```

---

## Hyperparameter Tuning

### 7. Adjust Learning Rate
```bash
python train.py --learning_rate 0.0005 --weight_decay 0.00005
```

### 8. Increase Model Capacity
```bash
python train.py --hidden_dim 128 --num_spatial_layers 3 --num_temporal_layers 3
```

### 9. Longer Sequences
```bash
python train.py --seq_len 12 --pred_len 6
```

### 10. Higher Dropout for Regularization
```bash
python train.py --dropout 0.3 --weight_decay 0.001
```

---

## Graph Construction Options

### 11. Distance-Based Graph
```bash
python train.py --graph_method distance --graph_distance_threshold 100
```

### 12. Denser KNN Graph
```bash
python train.py --graph_method knn --graph_k 16
```

---

## Production Training

### 13. Full Training Run (Recommended)
```bash
python train.py \
    --batch_size 2 \
    --hidden_dim 64 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --patience 20 \
    --dropout 0.2 \
    --graph_k 8
```

### 14. High-Capacity Model for Best Performance
```bash
python train.py \
    --batch_size 4 \
    --hidden_dim 128 \
    --num_spatial_layers 4 \
    --num_temporal_layers 4 \
    --num_heads 8 \
    --num_epochs 150 \
    --patience 25
```

---

## Custom Configuration

### 15. Use Custom Config File
```bash
python train.py --config my_config.json
```

### 16. Override Multiple Parameters
```bash
python train.py \
    --config config.json \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --num_epochs 50 \
    --seed 123
```

---

## Debugging & Development

### 17. Skip Visualization (Faster Debugging)
```bash
python train.py --num_epochs 1 --no_visualization
```

### 18. Reproducible Run
```bash
python train.py --seed 42 --batch_size 1
```

---

## Tips

- **Start with low batch_size (1-2)** if you encounter memory errors
- **Reduce hidden_dim** (32 or 64) for memory-constrained setups
- **Use --no_visualization** during development/debugging to speed up training
- **Increase patience** (20-30) for better model convergence
- **Monitor GPU memory** usage to find optimal batch_size for your hardware
- **Use smaller num_epochs** (5-10) for quick testing before full training

---

## Monitoring Training

After starting training, you can monitor progress in TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open `http://localhost:6006` in your browser.

---

## Check Results

After training completes, results are saved in:
- `checkpoints/best_model.pth` - Best model weights
- `results/training_history.json` - Training metrics
- `results/plots/*.png` - Visualization plots
- `results/performance_report.txt` - Summary report
