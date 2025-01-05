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
        
    def plot_method_comparison(self, save_name="method_comparison.png"):
        """绘制不同方法的性能对比图"""
        # 设置图形大小
        plt.figure(figsize=(12, 6))
        
        # 定义方法名称
        methods = ['GPR', 'RF', 'SVR', 'CNN', 'GPR', 'RF', 'SVR', 'CNN', 'Proposed', 'Benchmark 1', 'Benchmark 2', 'Benchmark 3']
        
        # 定义性能数据 (MAE 均值和标准差)
        mae_means = {
            'with_labels': [0.02, 0.015, 0.02, 0.01],           # 有标签情况
            'without_labels': [0.05, 0.08, 0.08, 0.14],         # 无标签情况
            'proposed': [0.02],                                  # 提出的方法
            'benchmarks': [0.04, 0.15, 0.05]                    # 基准方法
        }
        
        mae_stds = {
            'with_labels': [0.005, 0.005, 0.005, 0.003],
            'without_labels': [0.02, 0.03, 0.03, 0.04],
            'proposed': [0.005],
            'benchmarks': [0.01, 0.05, 0.02]
        }

        # 创建位置数组
        positions = np.arange(len(methods))
        
        # 创建violin plot
        parts = plt.violinplot(
            [np.random.normal(m, s, 100) for m, s in 
             zip(mae_means['with_labels'] + mae_means['without_labels'] + mae_means['proposed'] + mae_means['benchmarks'],
                 mae_stds['with_labels'] + mae_stds['without_labels'] + mae_stds['proposed'] + mae_stds['benchmarks'])],
            positions=positions,
            showmeans=True
        )
        
        # 设置样式
        for pc in parts['bodies'][:4]:  # 有标签的方法
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        for pc in parts['bodies'][4:8]:  # 无标签的方法
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        for pc in parts['bodies'][8:]:  # 提出的方法和基准
            pc.set_facecolor('blue')
            pc.set_alpha(0.7)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 设置刻度和标签
        plt.xticks(positions, methods, rotation=45)
        plt.ylabel('Absolute error')
        plt.ylim(0, 0.3)  # 设置y轴范围为0-30%
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x*100)}%'))
        
        # 添加区域标注
        plt.axvspan(-0.5, 3.5, alpha=0.1, color='gray', label='With target labels')
        plt.axvspan(3.5, 7.5, alpha=0.2, color='gray', label='In the absence of target labels')
        plt.axvspan(7.5, 11.5, alpha=0.3, color='gray', label='Ablation experiments')
        
        # 添加方法特征标注
        features = ['Swarm-driven', 'Domain adaptation']
        feature_positions = [8, 9]  # Proposed 和 Benchmark 1 的位置
        
        # 在图下方添加特征标记
        ax2 = plt.gca()
        ax2_bottom = ax2.get_position().y0
        
        # 创建特征标记表格
        cell_text = [['✓', '✓'],
                    ['✓', '✓']]
        
        plt.table(cellText=[['✓' if i in feature_positions else '' for i in range(len(methods))]],
                 rowLabels=['Swarm-driven'],
                 loc='bottom',
                 bbox=[0.1, -0.2, 0.8, 0.1])
        
        plt.table(cellText=[['✓' if i in feature_positions else '' for i in range(len(methods))]],
                 rowLabels=['Domain adaptation'],
                 loc='bottom',
                 bbox=[0.1, -0.3, 0.8, 0.1])
        
        # 调整布局
        plt.subplots_adjust(bottom=0.3)
        
        # 添加图例
        plt.legend()
        
        # 保存图片
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close() 