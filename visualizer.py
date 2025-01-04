import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import torch

class Visualizer:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_ensemble_analysis(self, models, input_size, save_name="ensemble_analysis.png"):
        """绘制集成模型分析图，类似论文中的图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 图 (a): 激活函数分析
        self._plot_activation_analysis(ax1)
        
        # 图 (b): 网络结构分析
        self._plot_architecture_analysis(ax2)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_activation_analysis(self, ax):
        """绘制不同激活函数的性能比较"""
        activations = ['ReLU', 'Tanh', 'Sigmoid', 'LogSigmoid']
        mae_means = [0.02, 0.17, 0.17, 0.40]  # 示例数据
        mae_stds = [0.01, 0.05, 0.05, 0.07]   # 示例数据
        
        # 创建violin plot
        parts = ax.violinplot([np.random.normal(m, s, 100) for m, s in zip(mae_means, mae_stds)],
                            positions=range(len(activations)))
        
        # 设置样式
        for pc in parts['bodies']:
            pc.set_facecolor('red')
            pc.set_alpha(0.3)
        
        ax.set_xticks(range(len(activations)))
        ax.set_xticklabels(activations)
        ax.set_xlabel('Activation function')
        ax.set_ylabel('Absolute error')
        ax.set_ylim(0, 0.8)  # 调整 Y 轴范围到 80%
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x*100)}%'))
        
    def _plot_architecture_analysis(self, ax):
        """绘制网络结构分析热力图"""
        layers = [1, 2, 3, 4]
        channels = [32, 64, 128, 256]
        
        # 创建性能矩阵（示例数据）
        performance = np.array([
            [0.09, 0.08, 0.07, 0.06],
            [0.08, 0.06, 0.05, 0.04],
            [0.07, 0.05, 0.03, 0.02],
            [0.06, 0.04, 0.02, 0.01]
        ])
        
        # 绘制热力图
        sns.heatmap(performance, 
                   annot=True, 
                   fmt='.2%',
                   cmap='Blues_r',
                   xticklabels=channels,
                   yticklabels=layers,
                   ax=ax)
        
        ax.set_xlabel('Number of channels per layer')
        ax.set_ylabel('Number of the CNN layer')
        
    def plot_soh_prediction(self, true_soh, pred_soh, save_name="soh_prediction.png"):
        """绘制SOH预测结果对比图"""
        plt.figure(figsize=(10, 6))
        
        # 绘制真实值和预测值
        plt.plot(range(len(true_soh)), true_soh, 
                 label='True SOH', color='blue', linewidth=2)
        plt.plot(range(len(pred_soh)), pred_soh, 
                 label='Predicted SOH', color='red', linestyle='--', linewidth=2)
        
        # 添加图例和标签
        plt.xlabel('Cycle Number')
        plt.ylabel('State of Health')
        plt.title('SOH Prediction Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_curves(self, train_losses, val_losses, save_name="training_curves.png"):
        """绘制训练过程中的损失曲线"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close() 