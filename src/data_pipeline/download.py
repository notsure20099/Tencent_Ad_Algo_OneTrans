import sys
import pandas as pd
import os
import requests
import json

# 配置Hugging Face国内镜像
hf_mirror = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = hf_mirror
os.environ["HUGGINGFACE_HUB_BASE_URL"] = hf_mirror

# 禁用所有代理环境变量
proxy_env_vars = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
for env_var in proxy_env_vars:
    if env_var in os.environ:
        del os.environ[env_var]

# 输出目录
data_dir = r"E:\py_project\Tencent_Ad_Algo_OneTrans\data\raw"
output_path = os.path.join(data_dir, "sample_data.parquet")

# 创建数据目录
os.makedirs(data_dir, exist_ok=True)

# 数据集信息
dataset_id = "TAAC2026/data_sample_1000"
file_name = "demo_1000.parquet"

# 检查是否已有数据文件
existing_file_path = os.path.join(data_dir, file_name)
if os.path.exists(existing_file_path):
    print(f"发现已有数据文件: {existing_file_path}")
    print("正在验证文件...")
    try:
        df = pd.read_parquet(existing_file_path)
        print(f"文件验证成功，数据形状: {df.shape}")
        
        # 如果output_path不同，复制文件
        if existing_file_path != output_path:
            print(f"复制文件到: {output_path}")
            df.to_parquet(output_path, index=False)
            print(f"文件已复制完成！")
        
        print("\n✓ 数据已准备好！")
        sys.exit(0)
    except Exception as e:
        print(f"文件验证失败: {type(e).__name__}: {e}")
        print("尝试重新下载...")

def download_with_huggingface_hub():
    """使用huggingface_hub下载数据"""
    print("尝试使用huggingface_hub下载数据...")
    try:
        # 使用pandas直接读取hf://协议的文件
        df = pd.read_parquet(f"hf://datasets/{dataset_id}/{file_name}")
        print(f"数据读取成功，形状: {df.shape}")
        
        # 保存数据
        df.to_parquet(output_path, index=False)
        print(f"数据已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"使用huggingface_hub下载失败: {type(e).__name__}: {e}")
        return False

def download_with_requests():
    """使用requests直接下载数据"""
    print("尝试使用requests直接下载数据...")
    try:
        # 构建下载URL
        download_url = f"{hf_mirror}/datasets/{dataset_id}/resolve/main/{file_name}"
        print(f"下载URL: {download_url}")
        
        # 发送请求
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # 保存文件
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"文件已下载到: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path)} 字节")
        
        # 验证文件
        df = pd.read_parquet(output_path)
        print(f"文件验证成功，数据形状: {df.shape}")
        return True
    except Exception as e:
        print(f"使用requests下载失败: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("开始下载数据集...")
    print(f"数据集: {dataset_id}")
    print(f"文件: {file_name}")
    print(f"镜像源: {hf_mirror}")
    print("=" * 50)
    
    # 尝试第一种方法
    if download_with_huggingface_hub():
        print("\n✓ 下载完成！")
        sys.exit(0)
    
    # 尝试第二种方法
    print("\n尝试备用方法...")
    if download_with_requests():
        print("\n✓ 下载完成！")
        sys.exit(0)
    
    # 如果都失败了
    print("\n✗ 所有下载方法都失败了！")
    print("\n请尝试手动下载：")
    print(f"1. 访问: https://huggingface.co/datasets/{dataset_id}")
    print(f"2. 下载文件: {file_name}")
    print(f"3. 保存到: {output_path}")
    sys.exit(1)