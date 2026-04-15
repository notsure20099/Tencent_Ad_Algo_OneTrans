import polars as pl

# 读取预处理前的数据
df = pl.read_parquet("data/raw/demo_1000.parquet")

# 统计分布
distribution = df["label_type"].value_counts().sort("count", descending=True)
print("Label Type 分布情况:")
print(distribution)

# 计算占比
total = len(df)
for row in distribution.to_dicts():
    percentage = (row['count'] / total) * 100
    print(f"类型 {row['label_type']}: {row['count']} 条 ({percentage:.2f}%)")