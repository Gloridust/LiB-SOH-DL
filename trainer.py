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
            for batch_idx, batch in enumerate(test_loader):
                try:
                    # 先解析 batch
                    if isinstance(batch, (tuple, list)):
                        if len(batch) == 2:
                            data_raw, target_raw = batch
                        else:
                            # 只有 data，没有 target
                            data_raw = batch[0] if isinstance(batch[0], (np.ndarray, list, torch.Tensor)) else batch
                            target_raw = None
                    else:
                        data_raw = batch
                        target_raw = None
                    
                    # 使用辅助函数转换 data
                    data_t = to_tensor_safely(data_raw, self.device)
                    if data_t is None:
                        self.logger.error(f"[eval] Batch {batch_idx} 数据转换失败，跳过。类型: {type(data_raw)}")
                        continue
                    
                    # 如果存在 target_raw，则同理进行转换
                    if target_raw is not None:
                        target_t = to_tensor_safely(target_raw, self.device)
                        if target_t is None:
                            self.logger.error(f"[eval] Batch {batch_idx} 标签转换失败，跳过。")
                            continue
                    else:
                        target_t = None
                    
                    # 进行推断
                    pred = model(data_t)
                    
                    # 收集预测值
                    predictions.extend(pred.cpu().numpy())
                    
                    # 如果有标签，则收集
                    if target_t is not None:
                        targets.extend(target_t.cpu().numpy())
                        
                except Exception as e:
                    self.logger.error(f"[eval] 第 {batch_idx} 个 batch 出错：{e}")
                    # 对 data_raw 进行更多 debug
                    self.logger.error(f"  data_raw 类型: {type(data_raw)}")
                    # data_raw 如果是 list, 可以再 print(len(data_raw)) 之类
                    continue
        
        # 如果没有收集到任何 target，则说明在目标域(无标签)，返回 None
        if len(targets) == 0:
            return None
        
        # 转换为 numpy 数组并计算指标
        try:
            predictions = np.array(predictions)
            targets = np.array(targets)
            
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

def to_tensor_safely(data, device):
    """
    将 data 安全地转换成 (batch_size, 1, length) 的 torch.Tensor 并放到指定 device。
    如果转换过程中出现任何异常，则返回 None。
    """
    try:
        # 如果已经是 np.ndarray，则直接用；否则先转为 np.array
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        # 如果是一维数据 (length, )，则视为 batch_size=1，channels=1
        if data.ndim == 1:
            data = data[None, None, :]  # (1, 1, length)
        elif data.ndim == 2:
            # 大多数情况下可能是 (batch_size, length)；再加一个通道维度
            data = data[:, None, :]     # (batch_size, 1, length)
        elif data.ndim == 3:
            # 假设是 (batch, channel, length)；若 channel != 1 看你需求
            pass
        else:
            # 其他维度暂不支持
            return None
        
        tensor_data = torch.tensor(data, dtype=torch.float32, device=device)
        return tensor_data
    except Exception:
        return None 