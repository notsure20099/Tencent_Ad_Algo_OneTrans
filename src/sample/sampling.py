import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import torch

# 1. 加载数据
path = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/raw/demo_1000.parquet'
df = pl.read_parquet(path)

#查看时间跨度，这只是一个13秒切片
print(df['timestamp'].min())
print(df['timestamp'].max())

def prepare_data(input_path, mapping_path, output_dir):
    # 1. 加载数据与映射表
    print(f"正在读取数据: {input_path}")
    df = pl.read_parquet(input_path)
    with open(mapping_path, 'r') as f:
        id_map = json.load(f)
    
    # 2. 确定特征列
    seq_cols = [col for col in df.columns if 'domain_' in col and 'seq' in col]
    target_col = 'label_type'
    
    # 3. 按时间序排序 (防止时空穿越)
    df = df.sort("timestamp")
    
    # 4. 划分训练集和验证集 (8:2)
    split_idx = int(len(df) * 0.8)
    train_df = df.head(split_idx)
    val_df = df.tail(len(df) - split_idx)
    
    print(f"数据划分完成: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条")

    # 5. 核心处理函数：将原始 ID 映射并转换为 Tensor
    def process_to_tensor(subset_df):
        all_features = []
        labels = subset_df[target_col].to_numpy()
        
        for i in range(len(subset_df)):
            row_feats = []
            for col in seq_cols:
                raw_seq = subset_df[col][i]
                
                # --- 修复逻辑开始 ---
                if raw_seq is None:
                    # 如果整个序列列为空，直接给全 0 填充
                    mapped = [0] * 128
                else:
                    # 正常的映射逻辑
                    mapped = [id_map.get(str(int(x)), 0) for x in raw_seq if x is not None]
                    # 截断并填充至 128 位
                    mapped = mapped[-128:] + [0] * (128 - len(mapped[-128:]))
                # --- 修复逻辑结束 ---
                
                row_feats.append(mapped)
            all_features.append(row_feats)
        
        return torch.tensor(all_features, dtype=torch.long), torch.tensor(labels, dtype=torch.float)
    # 6. 保存为 PyTorch 序列化文件 (加速训练读取)
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在转换训练集...")
    x_train, y_train = process_to_tensor(train_df)
    torch.save({'x': x_train, 'y': y_train}, os.path.join(output_dir, 'train_data.pt'))
    
    print("正在转换验证集...")
    x_val, y_val = process_to_tensor(val_df)
    torch.save({'x': x_val, 'y': y_val}, os.path.join(output_dir, 'val_data.pt'))
    
    print(f"✅ 预处理完成！文件已保存至 {output_dir}")

if __name__ == "__main__":
    prepare_data(
        input_path="/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/raw/demo_1000.parquet",
        mapping_path="/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/id_mapping.json",
        output_dir="/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/processed"
    )