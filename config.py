class Config:
    # 数据预处理参数
    VOLTAGE_WINDOW = 500  # mV
    VOLTAGE_INTERVAL = 10  # mV
    MIN_VOLTAGE = 3.0     # V
    MAX_VOLTAGE = 4.2     # V
    
    # 模型参数
    NUM_MODELS = 5  # 集成模型数量
    BATCH_SIZE = 16  # 减小批次大小
    LEARNING_RATE = 0.0001  # 降低学习率
    NUM_EPOCHS = 2000
    MIN_EPOCHS = 500
    
    # 训练参数
    DOMAIN_LOSS_WEIGHT = 0.1
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_THRESHOLD = 0.05
    PATIENCE = 20  # 增加早停耐心值
    
    # 优化器参数
    WEIGHT_DECAY = 1e-4  # L2正则化
    BETA1 = 0.9
    BETA2 = 0.999
    
    # 模型选择参数
    MODEL_SELECTION_THRESHOLD = 0.05
    
    # 设备配置
    @staticmethod
    def get_device():
        """获取可用的计算设备，按照 CUDA > MPS > CPU 的优先级"""
        import torch
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    DEVICE = get_device.__func__()  # 静态方法调用
    
    # 数据增强参数
    NOISE_LEVEL = 0.01 
    
    # 模型保存参数
    SAVE_INTERVAL = 100  # 每隔多少轮保存一次模型
    
    # 可视化参数
    PLOT_INTERVAL = 100  # 每隔多少轮绘制一次训练曲线 