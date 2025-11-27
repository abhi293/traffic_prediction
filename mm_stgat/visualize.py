"""
Visualization and results analysis for MM-STGAT
Creates comprehensive plots with bold labels and large font sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
from datetime import datetime
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Font settings for bold and enlarged labels
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 16
FONT_SIZE_TICK = 14
FONT_SIZE_LEGEND = 14

plt.rcParams.update({
    'font.size': FONT_SIZE_TICK,
    'axes.labelsize': FONT_SIZE_LABEL,
    'axes.titlesize': FONT_SIZE_TITLE,
    'xtick.labelsize': FONT_SIZE_TICK,
    'ytick.labelsize': FONT_SIZE_TICK,
    'legend.fontsize': FONT_SIZE_LEGEND,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold'
})


class ResultsVisualizer:
    """
    Comprehensive visualization suite for MM-STGAT results
    """
    def __init__(self, results_dir='results', output_dir='results/plots'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_results()
    
    def load_results(self):
        """Load all results"""
        print("Loading results...")
        
        # Training history
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'r') as f:
            self.history = json.load(f)
        
        # Test predictions
        predictions_path = os.path.join(self.results_dir, 'test_predictions.npz')
        data = np.load(predictions_path, allow_pickle=True)
        self.predictions = data['predictions']
        self.targets = data['targets']
        self.metrics = data['metrics'].item()
        
        print(f"✓ Loaded results")
        print(f"  Predictions shape: {self.predictions.shape}")
        print(f"  Targets shape: {self.targets.shape}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=FONT_SIZE_TITLE+4, fontweight='bold', y=0.995)
        
        epochs = np.arange(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'o-', label='Train Loss', 
                linewidth=2.5, markersize=6)
        ax.plot(epochs, self.history['val_loss'], 's-', label='Validation Loss', 
                linewidth=2.5, markersize=6)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Loss', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Training and Validation Loss', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # MAE curves
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_mae'], 'o-', label='Train MAE', 
                linewidth=2.5, markersize=6)
        ax.plot(epochs, self.history['val_mae'], 's-', label='Validation MAE', 
                linewidth=2.5, markersize=6)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('MAE', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Mean Absolute Error', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # Learning rate schedule
        ax = axes[1, 0]
        ax.plot(epochs, self.history['learning_rates'], 'D-', color='purple', 
                linewidth=2.5, markersize=6)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Learning Rate', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Learning Rate Schedule', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # Overfitting detection
        ax = axes[1, 1]
        gap = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        ax.plot(epochs, gap, 'o-', color='red', linewidth=2.5, markersize=6)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Val Loss - Train Loss', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Overfitting Gap', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_predictions_vs_actual(self, num_samples=5):
        """Plot predictions vs actual values"""
        batch_size, num_nodes, pred_len, output_dim = self.predictions.shape
        
        # Select random samples
        sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
        node_indices = np.random.choice(num_nodes, min(3, num_nodes), replace=False)
        
        fig, axes = plt.subplots(len(node_indices), len(sample_indices), 
                                figsize=(6*len(sample_indices), 5*len(node_indices)))
        
        if len(node_indices) == 1:
            axes = axes.reshape(1, -1)
        if len(sample_indices) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Predictions vs Actual Traffic Volume', 
                    fontsize=FONT_SIZE_TITLE+4, fontweight='bold', y=0.995)
        
        for i, node_idx in enumerate(node_indices):
            for j, sample_idx in enumerate(sample_indices):
                ax = axes[i, j]
                
                # Get predictions and targets for this sample and node
                pred = self.predictions[sample_idx, node_idx, :, 0]  # First feature (traffic volume)
                target = self.targets[sample_idx, node_idx, :, 0]
                
                time_steps = np.arange(1, pred_len + 1)
                
                ax.plot(time_steps, target, 'o-', label='Actual', 
                       linewidth=3, markersize=8, color='blue')
                ax.plot(time_steps, pred, 's-', label='Predicted', 
                       linewidth=3, markersize=8, color='red', alpha=0.7)
                
                ax.set_xlabel('Time Step', fontweight='bold', fontsize=FONT_SIZE_LABEL)
                ax.set_ylabel('Traffic Volume', fontweight='bold', fontsize=FONT_SIZE_LABEL)
                ax.set_title(f'Node {node_idx} | Sample {sample_idx}', 
                           fontweight='bold', fontsize=FONT_SIZE_TITLE-2)
                ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND-2, frameon=True)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=FONT_SIZE_TICK-2, width=2)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'predictions_vs_actual.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_error_distribution(self):
        """Plot error distribution"""
        # Calculate errors
        errors = self.predictions - self.targets
        mae_per_sample = np.abs(errors).mean(axis=(1, 2, 3))
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Error Analysis', fontsize=FONT_SIZE_TITLE+4, fontweight='bold', y=0.995)
        
        # Error histogram
        ax = axes[0, 0]
        ax.hist(errors.flatten(), bins=100, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Prediction Error', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Error Distribution', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=3)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # MAE per sample
        ax = axes[0, 1]
        ax.plot(mae_per_sample, linewidth=2.5)
        ax.set_xlabel('Sample Index', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('MAE', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('MAE per Sample', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # Error by time step
        ax = axes[1, 0]
        mae_per_timestep = np.abs(errors).mean(axis=(0, 1, 3))
        time_steps = np.arange(1, len(mae_per_timestep) + 1)
        ax.bar(time_steps, mae_per_timestep, edgecolor='black', linewidth=2, alpha=0.7)
        ax.set_xlabel('Prediction Time Step', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('MAE', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Error by Prediction Horizon', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.grid(True, alpha=0.3, linewidth=1.5, axis='y')
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(errors.flatten()[::100], dist="norm", plot=ax)
        ax.set_xlabel('Theoretical Quantiles', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Sample Quantiles', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Q-Q Plot (Normality Test)', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'error_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_scatter_predictions(self):
        """Scatter plot of predictions vs actual"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Prediction Accuracy', fontsize=FONT_SIZE_TITLE+4, fontweight='bold')
        
        # Flatten data
        pred_flat = self.predictions.flatten()
        target_flat = self.targets.flatten()
        
        # Sample for visualization (to avoid overcrowding)
        sample_size = min(10000, len(pred_flat))
        indices = np.random.choice(len(pred_flat), sample_size, replace=False)
        pred_sample = pred_flat[indices]
        target_sample = target_flat[indices]
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(target_sample, pred_sample, alpha=0.3, s=20, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(target_sample.min(), pred_sample.min())
        max_val = max(target_sample.max(), pred_sample.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Traffic Volume', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Predicted Traffic Volume', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Predictions vs Actual (Scatter)', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # Hexbin plot
        ax = axes[1]
        hb = ax.hexbin(target_sample, pred_sample, gridsize=50, cmap='YlOrRd', mincnt=1)
        ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=3, label='Perfect Prediction')
        
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Count', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        cb.ax.tick_params(labelsize=FONT_SIZE_TICK)
        
        ax.set_xlabel('Actual Traffic Volume', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Predicted Traffic Volume', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Density Plot', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'scatter_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_metrics_summary(self):
        """Plot summary of test metrics"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics_names = list(self.metrics.keys())
        metrics_values = list(self.metrics.values())
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax.bar(metrics_names, metrics_values, color=colors, 
                     edgecolor='black', linewidth=2.5, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        
        ax.set_ylabel('Metric Value', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Test Set Performance Metrics', fontweight='bold', fontsize=FONT_SIZE_TITLE+2)
        ax.grid(True, alpha=0.3, linewidth=1.5, axis='y')
        ax.tick_params(labelsize=FONT_SIZE_TICK+2, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.xticks(rotation=0, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'metrics_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive text report"""
        report_path = os.path.join(self.results_dir, 'performance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MM-STGAT TRAFFIC PREDICTION - PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total epochs: {len(self.history['train_loss'])}\n")
            f.write(f"Best validation loss: {min(self.history['val_loss']):.4f}\n")
            f.write(f"Final training loss: {self.history['train_loss'][-1]:.4f}\n")
            f.write(f"Final validation loss: {self.history['val_loss'][-1]:.4f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TEST SET PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            for metric, value in self.metrics.items():
                f.write(f"{metric.upper():15s}: {value:.4f}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("DATA STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Predictions shape: {self.predictions.shape}\n")
            f.write(f"  - Batch size: {self.predictions.shape[0]}\n")
            f.write(f"  - Number of nodes: {self.predictions.shape[1]}\n")
            f.write(f"  - Prediction horizon: {self.predictions.shape[2]}\n")
            f.write(f"  - Features: {self.predictions.shape[3]}\n\n")
            
            errors = self.predictions - self.targets
            f.write(f"Error statistics:\n")
            f.write(f"  Mean error: {errors.mean():.4f}\n")
            f.write(f"  Std error: {errors.std():.4f}\n")
            f.write(f"  Min error: {errors.min():.4f}\n")
            f.write(f"  Max error: {errors.max():.4f}\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"✓ Saved: {report_path}")
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80 + "\n")
        
        self.plot_training_curves()
        self.plot_predictions_vs_actual()
        self.plot_error_distribution()
        self.plot_scatter_predictions()
        self.plot_metrics_summary()
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
        print(f"\nAll plots saved to: {self.output_dir}")


def main():
    """Main visualization function"""
    visualizer = ResultsVisualizer()
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()
