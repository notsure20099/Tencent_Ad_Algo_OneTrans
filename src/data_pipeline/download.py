import polars as pl
df = pl.read_parquet("hf://datasets/TAAC2026/data_sample_1000/sample_data.parquet")
print(df.head())