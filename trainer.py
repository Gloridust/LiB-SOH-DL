import torch
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, config, model_dir="models"):
        self.config = config
        self.device = self._setup_device()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
    def _setup_device(self):
        """设置并返回计算设备"""
        device = torch.device(self.config.DEVICE)
        print(f"使用设备: {device}")
        
        # 如果使用CUDA，打印GPU信息
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
        # 如果使用MPS，打印相关信息
        elif device.type == 'mps':
            print("使用Apple Silicon GPU (MPS)")
        else:
            print("使用CPU")
            
        return device
    
    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.model_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_model(self, model, source_loader, target_loader, optimizer):
        """训练单个模型"""
        try:
            model = model.to(self.device)
            best_loss = float('inf')
            patience = self.config.PATIENCE
            patience_counter = 0
            
            train_losses = []
            val_losses = []
            
            for epoch in range(self.config.NUM_EPOCHS):
                model.train()
                total_loss = 0
                
                for (source_data, source_labels), target_data in zip(source_loader, target_loader):
                    try:
                        # 将数据移动到正确的设备上
                        source_data = source_data.to(self.device)
                        source_labels = source_labels.to(self.device)
                        target_data = target_data.to(self.device)
                        
                        optimizer.zero_grad()
                        
                        # 源域预测
                        source_features = model(source_data, extract_features=True)
                        source_pred = model(source_data)
                        
                        # 目标域预测
                        target_features = model(target_data, extract_features=True)
                        
                        # 计算损失
                        soh_loss = torch.nn.MSELoss()(source_pred, source_labels)
                        domain_loss = self._mmd_loss(source_features, target_features)
                        
                        loss = soh_loss + self.config.DOMAIN_LOSS_WEIGHT * domain_loss
                        
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        
                    except Exception as e:
                        self.logger.error(f"训练批次时出错: {str(e)}")
                        continue
                
                # 验证
                val_loss = self._validate(model, source_loader)
                train_losses.append(total_loss)
                val_losses.append(val_loss)
                
                # 早停检查
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self._save_model(model, epoch, val_loss)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience and epoch >= self.config.MIN_EPOCHS:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 100 == 0:
                    self.logger.info(f'Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], '
                                   f'Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')
                    
            return train_losses, val_losses
            
        except Exception as e:
            self.logger.error(f"训练过程出错: {str(e)}")
            raise
    
    def _validate(self, model, loader):
        """验证模型"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                pred = model(data)
                loss = torch.nn.MSELoss()(pred, labels)
                total_loss += loss.item()
                
        return total_loss
    
    def _mmd_loss(self, source_features, target_features):
        """计算MMD损失"""
        delta = source_features.mean(0) - target_features.mean(0)
        return torch.sum(delta * delta)
    
    def _save_model(self, model, epoch, loss):
        """保存模型"""
        save_path = self.model_dir / f"model_epoch_{epoch}_loss_{loss:.4f}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, save_path)
        
    def load_model(self, model, path):
        """加载模型"""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def plot_training_history(self, train_losses, val_losses, save_path=None):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                pred = model(data)
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
                
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算评估指标
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse)
        }
        
        # 保存评估结果
        results_path = self.model_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics 