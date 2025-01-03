class Config:
    # 数据预处理参数
    VOLTAGE_WINDOW = 500  # mV
    VOLTAGE_INTERVAL = 10  # mV
    MIN_VOLTAGE = 3.0     # V
    MAX_VOLTAGE = 4.2     # V
    
    # 模型参数
    NUM_MODELS = 5  # 集成模型数量
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 2000
    MIN_EPOCHS = 500
    
    # 训练参数
    DOMAIN_LOSS_WEIGHT = 0.1
    VALIDATION_SPLIT = 0.33
    EARLY_STOPPING_THRESHOLD = 0.05
    PATIENCE = 10  # 早停耐心值
    
    # 模型选择参数
    MODEL_SELECTION_THRESHOLD = 0.05
    
    # 设备配置
    DEVICE = 'cuda'  # 或 'cpu'
    
    # 数据增强参数
    NOISE_LEVEL = 0.01 
    
    # 模型保存参数
    SAVE_INTERVAL = 100  # 每隔多少轮保存一次模型
    
    # 可视化参数
    PLOT_INTERVAL = 100  # 每隔多少轮绘制一次训练曲线 