import torch
import time
import numpy as np
import math
import yaml
import os
from src.models.onetrans import OneTransModel

# --- 1. 环境与配置检查 ---
CONFIG_PATH = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/config/config.yaml'

def run_benchmark():
    # 强制打印第一行，确认程序已启动
    print("\n" + "="*50)
    print("探针启动：正在初始化推理延迟评估...")
    print("="*50)

    # 加载配置（用于获取 embed_dim 等参数）
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ 错误: 找不到配置文件 {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        CFG = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 当前评估设备: {device}")
    if device.type == 'cpu':
        print("⚠️ 警告: 未检测到 GPU，推理速度将极其缓慢（不具备比赛参考价值）")

    # --- 2. 模拟模型初始化 ---
    # 根据你的数据反馈，设置足够大的词表
    vocab_size = 2005000 
    seq_len = 1507
    model = OneTransModel(vocab_size, seq_len, CFG['model']).to(device)
    model.eval()

    # 准备输入数据 (BatchSize=1 为比赛标准评测)
    dummy_input = torch.randint(0, vocab_size, (1, seq_len)).to(device)

    # --- 3. 核心测量逻辑 ---
    warmups = 20
    repeats = 100
    
    print(f"🚀 开始热身 (Warmup x{warmups})...")
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(dummy_input)
    
    print(f"⏱️ 开始正式测量 (Repeats x{repeats})...")
    latencies = []
    
    with torch.no_grad():
        for i in range(repeats):
            # 关键：CUDA 同步
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # 执行推理
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
            if (i + 1) % 20 == 0:
                print(f"   已完成 {i+1}/{repeats} 次样本推理...")

    # --- 4. 统计结果输出 ---
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    std_dev = np.std(latencies)

    print("\n" + "📊 推理延迟评估报告 " + "="*30)
    print(f"🔹 平均延迟 (Mean):  {avg_latency:.4f} ms")
    print(f"🔹 99分位延迟 (P99): {p99_latency:.4f} ms")
    print(f"🔹 标准差 (Std Dev): {std_dev:.4f} ms")
    print(f"🔹 每秒吞吐量 (QPS): {1000 / avg_latency:.2f} items/s")
    print("="*50 + "\n")

# --- 关键：确保脚本被直接运行时会执行函数 ---
if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"❌ 运行中途崩溃: {e}")