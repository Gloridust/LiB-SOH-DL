import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class Visualizer:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_charging_curves(self, voltage_data, capacity_data, save_name="charging_curves.png"):
        """绘制充电曲线"""
        plt.figure(figsize=(10, 6))
        for i in range(min(len(voltage_data), 5)):  # 只画前5条曲线
            plt.plot(voltage_data[i], capacity_data[i], label=f'Cycle {i+1}')
            
        plt.xlabel('Voltage (V)')
        plt.ylabel('Capacity (Ah)')
        plt.title('Charging Curves')
        plt.legend()
        plt.savefig(self.save_dir / save_name)
        plt.close()
        
    def plot_soh_prediction(self, true_soh, pred_soh, save_name="soh_prediction.png"):
        """绘制SOH预测结果"""
        plt.figure(figsize=(10, 6))
        plt.scatter(true_soh, pred_soh, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # 理想预测线
        plt.xlabel('True SOH')
        plt.ylabel('Predicted SOH')
        plt.title('SOH Prediction')
        plt.savefig(self.save_dir / save_name)
        plt.close()
        
    def plot_error_distribution(self, errors, save_name="error_distribution.png"):
        """绘制误差分布"""
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        plt.savefig(self.save_dir / save_name)
        plt.close() 