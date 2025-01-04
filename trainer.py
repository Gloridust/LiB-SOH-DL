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
        
        # 创建检查点目录
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
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
        
    def save_checkpoint(self, model, optimizer, epoch, train_losses, val_losses, model_index):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        path = self.checkpoint_dir / f"checkpoint_model_{model_index}_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"保存检查点到: {path}")
        
    def load_checkpoint(self, model, optimizer, model_index):
        """加载最新的检查点"""
        try:
            checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_model_{model_index}_*.pt"))
            if not checkpoints:
                self.logger.info(f"没有找到模型 {model_index} 的检查点，从头开始训练")
                return model, 0, [], []
            
            latest_checkpoint = max(checkpoints, key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
            checkpoint = torch.load(latest_checkpoint)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.logger.info(f"成功加载检查点: {latest_checkpoint}")
            return model, checkpoint['epoch'], checkpoint['train_losses'], checkpoint['val_losses']
            
        except Exception as e:
            self.logger.warning(f"加载检查点时出错: {str(e)}，从头开始训练")
            return model, 0, [], []
    
    def train_model(self, model, source_loader, target_loader, optimizer, model_index):
        """训练单个模型"""
        try:
            model = model.to(self.device)
            start_epoch = 0
            train_losses = []
            val_losses = []
            
            # 如果配置了恢复训练，则尝试加载检查点
            if self.config.RESUME_TRAINING:
                model, start_epoch, train_losses, val_losses = self.load_checkpoint(
                    model, optimizer, model_index
                )
                if start_epoch > 0:
                    self.logger.info(f"从epoch {start_epoch} 恢复训练")
            
            best_loss = float('inf')
            patience = self.config.PATIENCE
            patience_counter = 0
            
            for epoch in range(start_epoch, self.config.NUM_EPOCHS):
                model.train()
                total_loss = 0
                
                for (source_data, source_labels), target_data in zip(source_loader, target_loader):
                    try:
                        # 将数据移动到正确的设备上并确保数据类型
                        source_data = source_data.to(self.device, dtype=torch.float32)
                        source_labels = source_labels.to(self.device, dtype=torch.float32)
                        target_data = target_data.to(self.device, dtype=torch.float32)
                        
                        optimizer.zero_grad()
                        
                        # 前向传播
                        source_pred = model(source_data)
                        loss = torch.nn.MSELoss()(source_pred, source_labels)
                        
                        # 反向传播
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    
                # 保存检查点
                if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        model, optimizer, epoch + 1,
                        train_losses, val_losses,
                        model_index
                    )
                
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
        """评估模型性能"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    target = None  # 初始化target变量
                    # 处理不同的数据格式
                    if isinstance(batch, tuple) and len(batch) == 2:
                        data, target = batch
                    else:
                        data = batch
                        if isinstance(data, tuple):
                            data = data[0]
                    
                    # 确保数据是张量
                    if isinstance(data, list):
                        data = torch.tensor(data, dtype=torch.float32)
                    if target is not None and isinstance(target, list):
                        target = torch.tensor(target, dtype=torch.float32)
                    
                    # 移动到设备并确保数据类型
                    data = data.to(self.device)
                    if target is not None:
                        target = target.to(self.device)
                    
                    # 预测
                    pred = model(data)
                    predictions.extend(pred.cpu().numpy())
                    
                    # 如果有目标值，添加到targets
                    if target is not None:
                        targets.extend(target.cpu().numpy())
                        
                except Exception as e:
                    self.logger.error(f"评估时出错: {str(e)}")
                    self.logger.error(f"数据类型: {type(data)}")
                    self.logger.error(f"数据形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
                    continue
        
        # 如果没有收集到targets（目标域评估），返回None
        if not targets:
            return None
        
        # 转换为numpy数组并计算指标
        try:
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
        except Exception as e:
            self.logger.error(f"计算评估指标时出错: {str(e)}")
            return None 