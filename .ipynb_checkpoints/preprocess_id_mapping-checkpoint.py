import polars as pl
import json
import os

def generate_mapping(input_path, output_path):
    print(f"正在读取数据: {input_path}")
    df = pl.read_parquet(input_path)
    
    # 获取所有序列特征列
    seq_cols = [col for col in df.columns if 'domain_' in col and 'seq' in col]
    
    print(f"开始提取 {len(seq_cols)} 个特征列的唯一 ID...")
    
    unique_ids = set()
    for col in seq_cols:
        # 使用 list.explode() 展开列表，并用 drop_nulls() 剔除空值
        col_uniques = (
            df.select(pl.col(col).list.explode())
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        # 再次确保 Python 层面没有 None
        unique_ids.update([x for x in col_uniques if x is not None])
    
    # 移除 0 (通常作为 Padding)
    if 0 in unique_ids:
        unique_ids.remove(0)
    
    print(f"去重完成，准备排序... 当前唯一值总数: {len(unique_ids)}")
    
    # 排序并映射
    sorted_ids = sorted(list(unique_ids))
    mapping = {int(old_id): i + 1 for i, old_id in enumerate(sorted_ids)}
    
    # 保存映射表
    with open(output_path, 'w') as f:
        json.dump(mapping, f)
    
    print(f"✅ 映射完成！")
    print(f"原始唯一 ID 总数: {len(sorted_ids)}")
    print(f"建议 Vocab Size: {len(sorted_ids) + 1} (包含 Padding 0)")
    print(f"映射表已保存至: {output_path}")
    return len(sorted_ids)

if __name__ == "__main__":
    RAW_DATA = "/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/raw/demo_1000.parquet"
    MAP_FILE = "/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/id_mapping.json"
    
    os.makedirs(os.path.dirname(MAP_FILE), exist_ok=True)
    generate_mapping(RAW_DATA, MAP_FILE)