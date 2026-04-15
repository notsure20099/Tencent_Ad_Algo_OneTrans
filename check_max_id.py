import pandas as pd
import numpy as np

# 加载数据
df = pd.read_parquet(r'E:\py_project\Tencent_Ad_Algo_OneTrans\data\raw\demo_1000.parquet')

print("检查序列特征中的最大ID值")
print("=" * 50)

# 获取所有序列特征列
domain_seq_cols = []
for col in df.columns:
    if 'domain_' in col and 'seq' in col:
        domain_seq_cols.append(col)

print(f"找到 {len(domain_seq_cols)} 个序列特征列")
print()

# 计算最大ID值
max_id = 0
for col in domain_seq_cols:
    print(f"处理列: {col}")
    try:
        # 获取非零值的最大值
        col_max = df[col].apply(lambda x: max(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 and max(x) != 0 else 0).max()
        print(f"  最大ID: {col_max}")
        if col_max > max_id:
            max_id = col_max
    except Exception as e:
        print(f"  错误: {e}")

print(f"\n所有序列特征中的最大ID值: {max_id}")
print(f"建议的词汇表大小: {max_id + 10000} (增加10000作为缓冲)")