import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

class Visualizer:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_ensemble_analysis(self, models, input_size, save_name="ensemble_analysis.png"):
        """绘制集成模型分析图，类似论文中的图"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 图 (a): 集成规模分析
        self._plot_ensemble_size_analysis(ax1, models)
        
        # 图 (b): 激活函数分析
        self._plot_activation_analysis(ax2)
        
        # 图 (c): 网络结构分析
        self._plot_architecture_analysis(ax3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_ensemble_size_analysis(self, ax, models):
        """绘制集成规模与MAE的关系"""
        sizes = [1, 50, 100, 150, 200, 250, 300]
        mae_means = []
        mae_stds = []
        
        for size in sizes:
            # 随机选择指定数量的模型
            selected_models = np.random.choice(models, min(size, len(models)), replace=True)
            predictions = []
            
            # 收集每个模型的预测结果
            for model in selected_models:
                try:
                    pred = model.predict()  # 使用新添加的predict方法
                    predictions.append(pred)
                except Exception as e:
                    print(f"模型预测出错: {str(e)}")
                    continue
            
            if not predictions:
                continue
            
            # 计算MAE及其标准差
            predictions = np.array(predictions).squeeze()
            mae = np.mean(np.abs(predictions - 0.5))  # 使用0.5作为基准值
            std = np.std(np.abs(predictions - 0.5))
            
            mae_means.append(mae)
            mae_stds.append(std)
        
        if mae_means:
            ax.fill_between(sizes[:len(mae_means)], 
                            np.array(mae_means) - np.array(mae_stds),
                            np.array(mae_means) + np.array(mae_stds),
                            alpha=0.2, color='red')
            ax.plot(sizes[:len(mae_means)], mae_means, 'k-', label='MAE')
            ax.set_xlabel('Size of DNN swarm')
            ax.set_ylabel('Absolute error')
            ax.set_ylim(0, 0.5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x*100)}%'))
            ax.legend()
        
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
        ax.set_ylim(0, 0.5)
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
        plt.scatter(range(len(true_soh)), true_soh, 
                   label='True SOH', alpha=0.6, color='blue')
        plt.scatter(range(len(pred_soh)), pred_soh, 
                   label='Predicted SOH', alpha=0.6, color='red')
        
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