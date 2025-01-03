import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def test_environment():
    """测试环境配置"""
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Pandas version:", pd.__version__)
    print("NumPy version:", np.__version__)
    
    # 测试CUDA可用性
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    # 测试目录结构
    required_dirs = ['dataset', 'models', 'visualizations', 'logs']
    for dir_name in required_dirs:
        path = Path(dir_name)
        if not path.exists():
            print(f"Creating directory: {dir_name}")
            path.mkdir(exist_ok=True)
        
    # 测试数据集文件
    dataset_files = ['Dataset#3.xlsx', 'Dataset#5.xlsx']
    for file_name in dataset_files:
        path = Path('dataset') / file_name
        if path.exists():
            print(f"Dataset file exists: {file_name}")
        else:
            print(f"Warning: Dataset file missing: {file_name}")

if __name__ == "__main__":
    test_environment() 