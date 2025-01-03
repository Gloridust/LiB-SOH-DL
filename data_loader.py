import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from scipy import interpolate

class BatteryDataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_excel_data(self, file_path):
        """加载Excel数据并预处理"""
        try:
            print(f"正在读取文件: {file_path}")
            df = pd.read_excel(
                file_path,
                header=1,
                engine='openpyxl'
            )
            
            # 获取电压列名（从3.0V到3.18V）
            voltage_columns = [col for col in df.columns if 'V' in str(col)]
            if not voltage_columns:
                raise ValueError("未找到电压列")
            
            # 过滤电压列，只保留3.0V到4.2V范围内的列
            voltage_values = []
            filtered_columns = []
            for col in voltage_columns:
                try:
                    v = float(col.replace('V', '').strip())
                    if 3.0 <= v <= 4.2:
                        voltage_values.append(v)
                        filtered_columns.append(col)
                except ValueError:
                    continue
            
            if not filtered_columns:
                raise ValueError("没有在有效范围内的电压列")
            
            print(f"找到的有效电压列: {filtered_columns}")
            
            # 提取样本数据
            samples = []
            for idx in range(len(df)):
                try:
                    # 获取充电容量曲线并转换为浮点数
                    capacity_curve = df.iloc[idx][filtered_columns].astype(float).values
                    
                    # 检查数据有效性
                    if np.any(pd.isna(capacity_curve)):
                        print(f"警告: 第{idx+1}行存在无效数据，将被跳过")
                        continue
                    
                    # 检查数据是否全为0或负值
                    if np.all(capacity_curve <= 0):
                        print(f"警告: 第{idx+1}行数据全为0或负值，将被跳过")
                        continue
                    
                    # 网格化处理
                    gridded_curve = self._grid_data(voltage_values, capacity_curve)
                    # 归一化处理
                    normalized_curve = self._normalize_curve(gridded_curve)
                    samples.append(normalized_curve)
                    
                except Exception as e:
                    print(f"警告: 处理第{idx+1}行数据时出错: {str(e)}")
                    continue
            
            if not samples:
                raise ValueError("没有有效的样本数据")
            
            print(f"成功加载 {len(samples)} 个样本")
            return np.array(samples, dtype=np.float32)
            
        except Exception as e:
            raise Exception(f"加载数据时出错: {str(e)}")
    
    def _grid_data(self, voltage, capacity):
        """网格化处理充电数据"""
        try:
            # 确保数据类型为float
            voltage = np.array(voltage, dtype=np.float32)
            capacity = np.array(capacity, dtype=np.float32)
            
            # 过滤掉异常电压值
            valid_mask = (voltage >= 3.0) & (voltage <= 4.2)
            if not np.any(valid_mask):
                raise ValueError("没有有效的电压值")
            
            voltage = voltage[valid_mask]
            capacity = capacity[valid_mask]
            
            # 按电压值排序
            sort_idx = np.argsort(voltage)
            voltage = voltage[sort_idx]
            capacity = capacity[sort_idx]
            
            # 创建插值函数
            f = interpolate.interp1d(
                voltage, 
                capacity, 
                kind='linear',
                bounds_error=False,  # 超出范围时返回nan而不是报错
                fill_value=np.nan    # 超出范围的值填充为nan
            )
            
            # 根据论文，使用10mV的电压间隔进行网格化
            voltage_window = self.config.VOLTAGE_WINDOW  # 500mV
            voltage_interval = self.config.VOLTAGE_INTERVAL  # 10mV
            
            # 创建网格点，确保在有效范围内
            start_v = max(3.0, min(voltage))
            end_v = min(4.2, start_v + voltage_window)
            
            grid_points = np.arange(
                start_v,
                end_v,
                voltage_interval,
                dtype=np.float32
            )
            
            # 对容量进行插值
            gridded_capacity = f(grid_points)
            
            # 检查插值结果是否有效
            if np.any(np.isnan(gridded_capacity)):
                raise ValueError("插值结果包含无效值")
            
            return gridded_capacity
            
        except Exception as e:
            print(f"网格化处理出错: {str(e)}")
            raise
    
    def _normalize_curve(self, capacity_curve):
        """归一化充电曲线"""
        try:
            # 确保数据类型为float
            capacity_curve = np.array(capacity_curve, dtype=np.float32)
            
            # 按最大容量进行归一化
            max_capacity = np.max(capacity_curve)
            if max_capacity == 0:
                return capacity_curve
            return capacity_curve / max_capacity
            
        except Exception as e:
            print(f"归一化处理出错: {str(e)}")
            raise
    
    def prepare_data(self, source_file, target_file):
        """准备源域和目标域数据"""
        # 加载数据
        source_data = self.load_excel_data(source_file)
        target_data = self.load_excel_data(target_file)
        
        # 计算SOH
        source_soh = self._calculate_soh(source_data)
        target_soh = self._calculate_soh(target_data)
        
        # 数据增强
        source_data, source_soh = self._augment_data(source_data, source_soh)
        
        return {
            'source': {
                'data': source_data,
                'soh': source_soh
            },
            'target': {
                'data': target_data,
                'soh': target_soh
            }
        }
    
    def _calculate_soh(self, data):
        """计算SOH
        SOH = 当前容量 / 初始容量
        """
        # 使用每个样本的最大容量
        capacities = np.max(data, axis=1)
        initial_capacity = capacities[0]  # 使用第一个循环的容量作为初始容量
        soh = capacities / initial_capacity
        return soh
    
    def _augment_data(self, data, soh, noise_level=0.01):
        """数据增强
        
        Args:
            data: 原始数据
            soh: SOH标签
            noise_level: 噪声水平
            
        Returns:
            augmented_data: 增强后的数据
            augmented_soh: 增强后的SOH标签
        """
        # 添加高斯噪声
        noise = np.random.normal(0, noise_level, data.shape)
        augmented_data = np.concatenate([data, data + noise])
        augmented_soh = np.concatenate([soh, soh])
        
        return augmented_data, augmented_soh
    
    def create_data_loaders(self, data_dict):
        """创建数据加载器"""
        # 创建源域数据集
        source_dataset = BatteryDataset(
            data_dict['source']['data'],
            data_dict['source']['soh'],
            is_source=True
        )
        
        # 创建目标域数据集
        target_dataset = BatteryDataset(
            data_dict['target']['data'],
            is_source=False
        )
        
        # 创建数据加载器
        source_loader = DataLoader(
            source_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        target_loader = DataLoader(
            target_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        return source_loader, target_loader

class BatteryDataset(Dataset):
    def __init__(self, voltage_curves, soh_labels=None, is_source=True):
        """
        Args:
            voltage_curves: 充电电压曲线数据
            soh_labels: SOH标签 (仅源域需要)
            is_source: 是否为源域数据
        """
        self.voltage_curves = torch.FloatTensor(voltage_curves)
        self.soh_labels = torch.FloatTensor(soh_labels).unsqueeze(1) if soh_labels is not None else None
        self.is_source = is_source
        
    def __len__(self):
        return len(self.voltage_curves)
    
    def __getitem__(self, idx):
        if self.is_source:
            return self.voltage_curves[idx], self.soh_labels[idx]
        return self.voltage_curves[idx]