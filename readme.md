# LiB-SOH-DL: 锂离子电池SOH预测深度学习模型

## 项目简介
本项目实现了一个基于深度学习的锂离子电池健康状态(State of Health, SOH)预测模型。该模型使用充电曲线数据进行训练，能够准确预测电池的剩余寿命。

## 特点
- 使用1D-CNN进行特征提取
- 集成多个模型提高预测稳定性
- 域适应技术处理不同工作条件
- 自动数据预处理和增强
- 支持多种计算设备(CPU/GPU/MPS)

## 环境要求
- Python 3.8+
- PyTorch 1.8+
- pandas
- numpy
- scipy
- matplotlib
- openpyxl

## 安装
1. 克隆仓库：
```bash
git clone https://github.com/yourusername/LiB-SOH-DL.git
cd LiB-SOH-DL
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
1. 在项目根目录下创建 `dataset` 文件夹
2. 将数据文件放入 `dataset` 文件夹：
   - Dataset#3.xlsx (源域数据)
   - Dataset#5.xlsx (目标域数据)

数据格式要求：
- Excel文件格式
- 第一行为表头
- 电压列名格式：`X.XX V`（如：3.00 V）
- 电压范围：3.0V - 4.2V

## 项目结构
```
LiB-SOH-DL/
├── dataset/                # 数据集目录
├── models/                 # 保存的模型
├── visualizations/         # 可视化结果
├── logs/                   # 训练日志
├── main.py                 # 主程序
├── config.py              # 配置文件
├── data_loader.py         # 数据加载器
├── trainer.py             # 训练器
├── visualizer.py          # 可视化工具
├── test.py                # 测试脚本
├── run.sh                 # 运行脚本
└── requirements.txt       # 依赖列表
```

## 使用方法

### 1. 配置参数
在 `config.py` 中设置模型参数：
```python
class Config:
    # 数据预处理参数
    VOLTAGE_WINDOW = 500    # mV
    VOLTAGE_INTERVAL = 10   # mV
    
    # 模型参数
    NUM_MODELS = 5         # 集成模型数量
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    
    # 训练参数
    NUM_EPOCHS = 2000
    PATIENCE = 20          # 早停耐心值
```

### 2. 运行训练
```bash
# 方法1：使用Python直接运行
python main.py

# 方法2：使用shell脚本（推荐）
chmod +x run.sh
./run.sh
```

### 3. 查看结果
训练结果将保存在以下目录：
- 模型文件：`models/`
- 训练曲线：`visualizations/training_history_*.png`
- 预测结果：`visualizations/final_prediction.png`
- 训练日志：`logs/training_*.log`

## 模型架构
1. 特征提取器（1D-CNN）：
   - 3层卷积层
   - ReLU激活函数
   - MaxPooling层

2. 中间全连接层（MFC）：
   - 2层全连接层
   - BatchNorm
   - Dropout

3. 终端全连接层（TFC）：
   - 2层全连接层
   - Sigmoid输出

## 评估指标
- MSE（均方误差）
- MAE（平均绝对误差）
- RMSE（均方根误差）

## 注意事项
1. 确保数据文件格式正确
2. 检查GPU内存是否充足
3. 适当调整batch_size
4. 观察训练日志及时发现问题

## 故障排除
1. 数据加载错误：
   - 检查数据文件路径
   - 确认Excel文件格式
   - 验证电压列名格式

2. 内存错误：
   - 减小batch_size
   - 减少模型数量
   - 使用数据采样

3. 训练不稳定：
   - 调整学习率
   - 增加早停耐心值
   - 检查数据预处理

## 贡献指南
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证
MIT License