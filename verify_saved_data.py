import os
import torch

# 检查保存的数据文件
save_dir = r'E:\py_project\Tencent_Ad_Algo_OneTrans\data\processed'

print("验证保存的数据文件")
print("=" * 50)

# 检查文件是否存在
files_to_check = [
    'ns_tokens.pt',
    'dense_tokens.pt',
    'sequence_tokens.pt'
]

for filename in files_to_check:
    file_path = os.path.join(save_dir, filename)
    if os.path.exists(file_path):
        print(f"✓ {filename} 存在")
        print(f"   文件大小: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
        
        # 尝试读取文件
        try:
            data = torch.load(file_path)
            print(f"   数据类型: {type(data)}")
            if isinstance(data, dict):
                print(f"   包含键: {list(data.keys())}")
                # 打印部分数据形状
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: 形状 {value.shape}")
        except Exception as e:
            print(f"   读取失败: {e}")
    else:
        print(f"✗ {filename} 不存在")
    print()

print("验证完成")