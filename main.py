import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import Config
from data_loader import BatteryDataLoader
from trainer import Trainer
from visualizer import Visualizer

# 定义深度神经网络模型
class BatterySOHEstimator(nn.Module):
    def __init__(self, input_size, device):
        super(BatterySOHEstimator, self).__init__()
        self.device = device
        self.input_size = input_size
        
        # 打印输入大小
        print(f"初始化网络，输入大小: {input_size}")
        
        # 计算每层卷积后的特征大小
        def calc_conv_output_size(input_size, kernel_size=3, stride=1, padding=1):
            return (input_size + 2 * padding - kernel_size) // stride + 1
            
        def calc_pool_output_size(input_size, kernel_size=2, stride=2):
            return input_size // stride
        
        # 第一层卷积+池化后的大小
        conv1_size = calc_conv_output_size(input_size)
        pool1_size = calc_pool_output_size(conv1_size)
        
        # 第二层卷积+池化后的大小
        conv2_size = calc_conv_output_size(pool1_size)
        pool2_size = calc_pool_output_size(conv2_size)
        
        # 第三层卷积后的大小
        conv3_size = calc_conv_output_size(pool2_size)
        
        print(f"网络结构大小:")
        print(f"输入 -> {input_size}")
        print(f"Conv1 -> {conv1_size} -> Pool1 -> {pool1_size}")
        print(f"Conv2 -> {conv2_size} -> Pool2 -> {pool2_size}")
        print(f"Conv3 -> {conv3_size}")
        
        # 1D CNN特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层: 卷积 + 池化
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第二层: 卷积 + 池化
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第三层: 卷积
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 初始化权重
        self._initialize_weights()
        
        # 计算展平后的特征维度
        with torch.no_grad():
            x = torch.randn(2, 1, self.input_size, dtype=torch.float32)  # 明确指定数据类型
            x = self.feature_extractor(x)
            self.feature_size = x.view(x.size(0), -1).size(1)
            print(f"展平后的特征大小: {self.feature_size}")
        
        # 中间全连接层 (MFC)
        self.mfc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 终端全连接层 (TFC)
        self.tfc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, extract_features=False):
        try:
            # 确保输入维度和数据类型正确
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = x.to(dtype=torch.float32)  # 明确指定数据类型
            
            # 特征提取
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            
            # MFC层
            mfc_output = self.mfc(features)
            
            if extract_features:
                return mfc_output
                
            # TFC层
            soh_pred = self.tfc(mfc_output)
            return soh_pred
            
        except Exception as e:
            print(f"前向传播错误: {str(e)}")
            print(f"输入张量形状: {x.shape}")
            print(f"输入张量类型: {x.dtype}")
            print(f"输入张量设备: {x.device}")
            raise
    
    def predict(self, data=None):
        """预测方法"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            if data is not None:
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                if len(data.shape) == 2:
                    data = data.unsqueeze(1)
                data = data.to(self.device)
                return self(data).cpu().numpy()
            else:
                # 如果没有提供数据，返回一个示例预测
                x = torch.randn(1, 1, self.input_size, device=self.device)
                return self(x).cpu().numpy()

# MMD损失函数
def mmd_loss(source_features, target_features):
    """计算最大均值差异损失"""
    delta = source_features.mean(0) - target_features.mean(0)
    return torch.sum(delta * delta)

# 主训练函数
def train_model(source_loader, target_loader, model, optimizer, num_epochs=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for (source_data, source_labels), target_data in zip(source_loader, target_loader):
            source_data = source_data.to(device).float()
            source_labels = source_labels.to(device).float()
            target_data = target_data.to(device).float()
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 源域预测
            source_features = model(source_data, extract_features=True)
            source_pred = model(source_data)
            
            # 目标域预测
            target_features = model(target_data, extract_features=True)
            
            # 计算损失
            soh_loss = nn.MSELoss()(source_pred, source_labels)
            domain_loss = mmd_loss(source_features, target_features)
            
            # 总损失
            loss = soh_loss + 0.1 * domain_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 打印训练进度
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

# 数据预处理函数
class DataPreprocessor:
    def __init__(self, voltage_window=500, voltage_interval=10):
        """
        参数:
            voltage_window: 电压采样窗口 (mV)
            voltage_interval: 网格化间隔 (mV)
        """
        self.voltage_window = voltage_window
        self.voltage_interval = voltage_interval
        
    def normalize_curve(self, voltage_curve, capacity_curve):
        """归一化充电曲线"""
        # 按名义容量归一化
        nominal_capacity = capacity_curve[0]  # 使用首次循环容量作为名义容量
        normalized_capacity = capacity_curve / nominal_capacity
        return voltage_curve, normalized_capacity
    
    def grid_data(self, voltage, current):
        """网格化处理充电数据"""
        # 创建电压网格点
        grid_points = np.arange(
            voltage.min(), 
            min(voltage.max(), voltage.min() + self.voltage_window),
            self.voltage_interval
        )
        
        # 对电流进行插值
        gridded_current = np.interp(grid_points, voltage, current)
        return grid_points, gridded_current

class EnsembleDNNs:
    def __init__(self, input_size, num_models=5, device=None):
        """
        参数:
            input_size: 输入特征大小
            num_models: 集成的模型数量
        """
        self.models = [BatterySOHEstimator(input_size, device) for _ in range(num_models)]
        self.optimizers = [
            optim.Adam(
                model.parameters(), 
                lr=Config.LEARNING_RATE,
                betas=(Config.BETA1, Config.BETA2),
                weight_decay=Config.WEIGHT_DECAY
            ) for model in self.models
        ]
        
    def train_all(self, source_loader, target_loader, num_epochs=2000):
        """训练所有模型"""
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            print(f"Training model {i+1}/{len(self.models)}")
            train_model(source_loader, target_loader, model, optimizer, num_epochs)
            
    def select_best_models(self, validation_loader, threshold=0.05):
        """选择最佳模型"""
        predictions = []
        for model in self.models:
            model.eval()
            model_preds = []
            with torch.no_grad():
                for batch in validation_loader:
                    data = batch[0]
                    pred = model(data.float())
                    model_preds.extend(pred.cpu().numpy())
            predictions.append(model_preds)
            
        predictions = np.array(predictions)
        means = predictions.mean(axis=1)
        stds = predictions.std(axis=1)
        
        # 选择标准差小于阈值的模型
        selected_indices = np.where(stds < threshold)[0]
        self.selected_models = [self.models[i] for i in selected_indices]
        return self.selected_models
    
    def predict(self, data_loader):
        """使用选定的模型进行预测"""
        predictions = []
        for model in self.selected_models:
            model.eval()
            model_preds = []
            with torch.no_grad():
                for batch in data_loader:
                    if isinstance(batch, tuple):
                        batch = batch[0]
                    pred = model(batch.float())
                    model_preds.extend(pred.cpu().numpy())
            predictions.append(model_preds)
        
        # 返回平均预测值
        return np.mean(predictions, axis=0)

def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            pred = model(data.float())
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算评估指标
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }

def main():
    # 加载配置
    config = Config()
    
    # 初始化数据加载器
    data_loader = BatteryDataLoader(config)
    
    # 准备数据
    data_dict = data_loader.prepare_data(
        source_file='./dataset/Dataset#3.xlsx',
        target_file='./dataset/Dataset#5.xlsx'
    )
    
    # 创建数据加载器
    source_loader, target_loader = data_loader.create_data_loaders(data_dict)
    
    # 获取输入特征大小
    input_size = next(iter(source_loader))[0].shape[-1] # 从数据加载器中获取正确的输入大小
    
    # 获取设备信息
    device = torch.device(config.DEVICE)
    
    # 创建模型集成
    ensemble = EnsembleDNNs(input_size, num_models=config.NUM_MODELS, device=device)
    
    # 初始化训练器和可视化器
    trainer = Trainer(config)
    visualizer = Visualizer()
    
    # 训练模型
    for i, (model, optimizer) in enumerate(zip(ensemble.models, ensemble.optimizers)):
        print(f"\nTraining model {i+1}/{len(ensemble.models)}")
        train_losses, val_losses = trainer.train_model(
            model, 
            source_loader, 
            target_loader, 
            optimizer,
            model_index=i+1  # 添加模型索引
        )
        
        # 绘制训练历史
        trainer.plot_training_history(
            train_losses, 
            val_losses,
            save_path=f"visualizations/training_history_model_{i+1}.png"
        )
    
    # 选择最佳模型
    selected_models = ensemble.select_best_models(
        source_loader,
        threshold=config.MODEL_SELECTION_THRESHOLD
    )
    
    # 进行预测
    predictions = ensemble.predict(target_loader)
    
    # 评估模型
    for i, model in enumerate(selected_models):
        print(f"\nModel {i+1} Performance:")
        # 源域评估
        source_metrics = trainer.evaluate_model(model, source_loader)
        if source_metrics:
            print("Source Domain Performance:")
            for metric_name, value in source_metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        # 目标域评估
        target_metrics = trainer.evaluate_model(model, target_loader)
        if target_metrics:
            print("Target Domain Performance:")
            for metric_name, value in target_metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
    # 创建可视化器
    visualizer = Visualizer()
    
    # 绘制方法对比图
    visualizer.plot_method_comparison(save_name="method_comparison.png")
    
    # 训练完成后进行分析
    visualizer.plot_ensemble_analysis(
        ensemble.models,
        input_size,
        save_name="ensemble_analysis.png"
    )
    
    # 绘制训练曲线
    visualizer.plot_training_curves(
        train_losses,
        val_losses,
        save_name="training_curves.png"
    )
    
    # 绘制预测结果
    visualizer.plot_soh_prediction(
        data_dict['target']['soh'],
        predictions,
        save_name="soh_prediction.png"
    )

if __name__ == "__main__":
    main()
