import polars as pl
import json
import os

def generate_mapping(input_path, output_path):
    df = pl.read_parquet(input_path)
    
    # 修复 1：同时提取序列 (s) 和非序列 (ns) 特征列
    # 假设你的配置文件中定义了所有需要映射的列名
    target_cols = [col for col in df.columns if 'domain_' in col] 
    
    unique_ids = set()
    for col in target_cols:
        dtype = df.schema[col]
        if isinstance(dtype, pl.List):
            # 处理序列
            col_uniques = df.select(pl.col(col).list.explode()).drop_nulls().unique().to_series().to_list()
        else:
            # 处理标量
            col_uniques = df.select(pl.col(col)).drop_nulls().unique().to_series().to_list()
        
        unique_ids.update([x for x in col_uniques if x is not None and x != 0])
    
    sorted_ids = sorted(list(unique_ids))
    
    # 修复 2：预留 0 为 PAD, 1 为 SEP，映射从 2 开始
    # 这样可以彻底避免特征 ID 与特殊功能 Token 冲突
    mapping = {int(old_id): i + 2 for i, old_id in enumerate(sorted_ids)}
    
    with open(output_path, 'w') as f:
        json.dump(mapping, f)
    
    # 修复 3：Vocab Size 必须是 最大索引 + 1
    max_index = len(sorted_ids) + 1 # 因为从 2 开始，共有 len 个，最大就是 len+1
    print(f"✅ 映射完成！")
    print(f"建议模型 Vocab Size 设置为: {max_index + 1}") 
    return max_index + 1

if __name__ == "__main__":
    RAW_DATA = "/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/raw/demo_1000.parquet"
    MAP_FILE = "/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/id_mapping.json"
    
    os.makedirs(os.path.dirname(MAP_FILE), exist_ok=True)
    generate_mapping(RAW_DATA, MAP_FILE)