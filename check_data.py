import pandas as pd
import numpy as np
from pathlib import Path

def validate_data(df, voltage_columns):
    """验证数据的有效性"""
    print("\n数据验证:")
    
    # 检查数据类型
    print("\n数据类型:")
    print(df[voltage_columns].dtypes)
    
    # 检查是否有非数值数据
    non_numeric = df[voltage_columns].applymap(lambda x: not pd.api.types.is_numeric_dtype(type(x)))
    if non_numeric.any().any():
        print("\n警告: 发现非数值数据:")
        for col in voltage_columns:
            non_numeric_rows = df[~pd.to_numeric(df[col], errors='coerce').notnull()]
            if not non_numeric_rows.empty:
                print(f"列 {col} 中的非数值数据:")
                print(non_numeric_rows[col])
    
    # 检查异常值
    print("\n数值范围检查:")
    for col in voltage_columns:
        col_data = pd.to_numeric(df[col], errors='coerce')
        print(f"\n{col}:")
        print(f"最小值: {col_data.min()}")
        print(f"最大值: {col_data.max()}")
        print(f"平均值: {col_data.mean()}")
        print(f"标准差: {col_data.std()}")

def check_excel_file(file_path):
    """检查Excel文件的结构和内容"""
    print(f"\n检查文件: {file_path}")
    
    try:
        # 尝试读取文件
        df = pd.read_excel(file_path, header=1, engine='openpyxl')
        print("文件读取成功")
        
        # 检查基本信息
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 检查电压列
        voltage_columns = [col for col in df.columns if 'V' in str(col)]
        print(f"找到 {len(voltage_columns)} 个电压列")
        
        # 验证数据
        validate_data(df, voltage_columns)
        
    except Exception as e:
        print(f"错误: {str(e)}")

def main():
    # 检查数据集目录
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("创建dataset目录")
        dataset_dir.mkdir(exist_ok=True)
    
    # 检查数据文件
    files_to_check = ['Dataset#3.xlsx', 'Dataset#5.xlsx']
    for file_name in files_to_check:
        file_path = dataset_dir / file_name
        if file_path.exists():
            check_excel_file(file_path)
        else:
            print(f"\n警告: 找不到文件 {file_name}")
            print(f"请确保文件位于: {file_path}")

if __name__ == "__main__":
    main() 