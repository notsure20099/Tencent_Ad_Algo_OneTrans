import pandas as pd
import numpy as np

# 加载数据
df = pd.read_parquet(r'E:\py_project\Tencent_Ad_Algo_OneTrans\data\raw\demo_1000.parquet')

print("数据基本信息:")
print(df.info())
print()

print("时间戳字段示例:")
print(df['timestamp'].head(5))
print()

print("序列特征示例:")
# 直接使用已知的序列特征列名
domain_seq_cols = {
    'domain_a': 'domain_a_seq_38',
    'domain_b': 'domain_b_seq_67',
    'domain_c': 'domain_c_seq_27',
    'domain_d': 'domain_d_seq_17'
}

for domain, col in domain_seq_cols.items():
    if col in df.columns:
        print(f"{col}: {df[col].iloc[0][:10]}...")
        print(f"  数据类型: {type(df[col].iloc[0])}")
        if isinstance(df[col].iloc[0], (list, np.ndarray)):
            print(f"  长度: {len(df[col].iloc[0])}")
        print(f"  非零值数量: {sum(1 for x in df[col].iloc[0] if x != 0)}")
print()

print("所有时间戳相关字段:")
print([col for col in df.columns if 'timestamp' in col])
print()

print("每个域的序列特征数量:")
domain_counts = {
    'domain_a': len([col for col in df.columns if 'domain_a_seq' in col]),
    'domain_b': len([col for col in df.columns if 'domain_b_seq' in col]),
    'domain_c': len([col for col in df.columns if 'domain_c_seq' in col]),
    'domain_d': len([col for col in df.columns if 'domain_d_seq' in col])
}
print(domain_counts)

print("\n序列特征列名示例:")
print("Domain A:", [col for col in df.columns if 'domain_a_seq' in col][:3])
print("Domain B:", [col for col in df.columns if 'domain_b_seq' in col][:3])
print("Domain C:", [col for col in df.columns if 'domain_c_seq' in col][:3])
print("Domain D:", [col for col in df.columns if 'domain_d_seq' in col][:3])