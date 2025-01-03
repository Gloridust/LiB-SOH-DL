#!/bin/bash

# 检查Python环境
python --version

# 创建虚拟环境（如果需要）
# python -m venv venv
# source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 创建必要的目录
mkdir -p dataset models visualizations logs

# 检查数据
echo "检查数据文件..."
python check_data.py

# 如果数据检查通过，运行训练
if [ $? -eq 0 ]; then
    echo "开始训练..."
    python main.py
else
    echo "数据检查失败，请检查数据文件"
    exit 1
fi 