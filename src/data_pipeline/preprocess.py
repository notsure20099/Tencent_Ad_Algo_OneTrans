import pandas as pd
import numpy as np
import json
import json
from collections import Counter

def clean_item_id(x):
    if pd.isna(x):
        return 0 # 或者根据业务逻辑处理缺失值
    # 如果是字符串，移除逗号；如果是数值，直接转换
    if isinstance(x, str):
        return int(x.replace(',', ''))
    return int(x)

def clean_timestamp(x):
    if pd.isna(x):
        return 0 # 或者根据业务逻辑处理缺失值
    # 如果是字符串，移除逗号；如果是数值，直接转换
    if isinstance(x, str):
        return int(x.replace(',', ''))
    return int(x)

def preprocess_multi_features(df, column_name='item_feature'):
    """
    工业级处理函数：解析 item_feature 并自动填充缺失值
    """
    
    # 1. 定义单行处理逻辑 (注意：这里只接收行数据 row，不操作 df)
    def extract_row_logic(row):
        # 1. 处理 pd.NA, None 或 np.nan
        if row is None or row.size == 0:
            return {}
    
        # 2. 统一处理：如果是字符串，尝试解析为列表
        items = row
        if isinstance(row, str):
            if row == "" or row == "[]":
                return {}
            try:
                items = json.loads(row)
            except (json.JSONDecodeError, TypeError):
                return {}
    
        # 3. 此时 items 应该是列表类型。检查是否为空列表
        # 使用 len() 判断，避开 row == "[]" 的逻辑错误
        if not isinstance(items, (list, np.ndarray)) or len(items) == 0:
            return {}

        res = {}
        for item in items:
            # 确保 item 是字典（防止数据中夹杂非字典元素）
            if not isinstance(item, dict):
                continue
            
            f_id = item.get('feature_id')
            v_type = item.get('feature_value_type')
        
            if f_id is not None and v_type is not None:
                res[f'feat_{f_id}'] = item.get(v_type)
        return res

    print(f"开始解析字段: {column_name} ...")
    
    # 2. 核心操作：apply 得到的是一个由 dict 组成的 Series，再用 apply(pd.Series) 展开
    extracted_df = df[column_name].apply(extract_row_logic).apply(pd.Series)
    
    # 4. 合并回主表
    result_df = pd.concat([df, extracted_df], axis=1)

    print("Item 特征处理完成。")
    return result_df

def process_label(label_str):
    labels = label_str[0]
    res = str(labels.get('action_type', 0)) + '_' + str(labels.get('action_time', 0))
    return res

if __name__ == "__main__":
    # 加载原始数据
    print("加载数据文件...")
    df = pd.read_parquet(r"E:\py_project\Tencent_Ad_Algo_OneTrans\data\raw\sample_data.parquet")
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据列: {df.columns.tolist()}")

    processed_df = preprocess_multi_features(df)
    processed_df = preprocess_multi_features(processed_df, column_name='user_feature')
    processed_df['item_id_processed'] = df['item_id'].apply(clean_item_id)
    processed_df['user_id_processed'] = df['user_id']
    processed_df['timestamp_processed'] = df['timestamp'].apply(clean_timestamp)
    processed_df['label_processed'] = df['label'].apply(process_label)
    
    print("--- item_id 处理结果对比 (前3行) ---")
    print(df.columns) 
    print("--- item_feature 处理结果对比 (前3行) ---")
    print(processed_df.columns) 

    print("--- label 处理结果对比 (前3行) ---")
    print(processed_df['label_processed'].head(3))