import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载数据
path = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/raw/demo_1000.parquet'
df = pl.read_parquet(path)

print(df.columns)
# 2. 提取各列的唯一 ID 集合
seq_cols = [col for col in df.columns if 'domain_' in col and 'seq' in col]
col_id_sets = {}

print("正在提取各列 ID 集合...")
for col in seq_cols:
    # 展开并去重，过滤掉 0 和 null
    ids = df.select(pl.col(col).list.explode()).drop_nulls().unique().to_series().to_list()
    ids_set = set([x for x in ids if x != 0])
    if len(ids_set) > 0:
        col_id_sets[col] = ids_set

# 3. 计算两两之间的交集比例
names = list(col_id_sets.keys())
n = len(names)
overlap_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

print(f"正在计算 {n}x{n} 交叉重合度...")
for i in range(n):
    for j in range(i, n):
        set_i = col_id_sets[names[i]]
        set_j = col_id_sets[names[j]]
        
        intersection = len(set_i.intersection(set_j))
        union = len(set_i.union(set_j))
        
        # 计算 Jaccard 相似度
        similarity = intersection / union if union > 0 else 0
        overlap_matrix.iloc[i, j] = similarity
        overlap_matrix.iloc[j, i] = similarity

# 4. 打印重合度较高的列对
print("\n[高重合度特征对 (Similarity > 0.05)]")
print("-" * 50)
for i in range(n):
    for j in range(i + 1, n):
        sim = overlap_matrix.iloc[i, j]
        if sim > 0.05:
            print(f"{names[i]} <--> {names[j]}: {sim:.4f}")

# 5. 可视化（如果 DSW 支持显示）
# 如果在终端运行，可以看上面的文字输出