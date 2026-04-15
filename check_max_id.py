import pandas as pd
import numpy as np

# 加载数据
data_path = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/raw/demo_1000.parquet'
df = pd.read_parquet(data_path)

print(f"检查序列特征统计信息 | 数据量: {len(df)}")
print("=" * 60)

# 获取所有序列特征列
domain_seq_cols = [col for col in df.columns if 'domain_' in col and 'seq' in col]
print(f"找到 {len(domain_seq_cols)} 个序列特征列")
print(f"{'列名':<25} | {'最大ID':<12} | {'唯一值数量':<12} | {'稀疏度/建议'}")
print("-" * 60)

global_max_id = 0
summary_stats = []

for col in domain_seq_cols:
    try:
        # 1. 展平数据：将 Series of Lists 转换为一维 Array
        # 去掉空值和空列表，提升效率
        all_elements = np.concatenate([x for x in df[col] if isinstance(x, (list, np.ndarray)) and len(x) > 0])
        
        # 2. 计算最大值
        col_max = all_elements.max() if len(all_elements) > 0 else 0
        
        # 3. 计算唯一值数量
        unique_count = len(np.unique(all_elements)) if len(all_elements) > 0 else 0
        
        # 4. 判定建议
        # 如果 最大ID 远大于 唯一值数量，说明中间有大量空洞，必须 Re-index
        suggestion = "需重映射" if col_max > unique_count * 2 else "较连续"
        if col_max > 1_000_000:
            suggestion += " (高维)"

        print(f"{col:<25} | {col_max:<12} | {unique_count:<12} | {suggestion}")

        if col_max > global_max_id:
            global_max_id = col_max
            
        summary_stats.append({
            'col': col,
            'max_id': col_max,
            'unique_count': unique_count
        })
        
    except Exception as e:
        print(f"{col:<25} | 错误: {str(e)[:20]}")

print("=" * 60)
print(f"全局最大 ID: {global_max_id}")
print(f"建议词汇表大小 (若不重映射): {global_max_id + 10000}")

# 统计总唯一值（假设所有特征共享一个 Embedding 空间）
total_unique = sum([s['unique_count'] for s in summary_stats])