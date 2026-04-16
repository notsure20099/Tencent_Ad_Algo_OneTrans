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
    print("\n" + "="*50)
    print("🚀 启动 OneTrans 预测延迟全局评估...")
    print("="*50)

    # 加载配置
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ 错误: 找不到配置文件 {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        CFG = yaml.safe_load(f)
    
    # 自动设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 当前评估设备: {device}")
    
    # 动态获取 Vocab Size (从处理好的数据中获取，最稳妥)
    path_train = os.path.join(CFG['paths']['processed_dir'], 'train_data.pt')
    if os.path.exists(path_train):
        train_data = torch.load(path_train, weights_only=True)
        vocab_size = train_data['x'].max().item() + 1
        seq_len = train_data['x'].shape[1]
    else:
        # 兜底方案
        vocab_size = 2005000 
        seq_len = 1507
        print("⚠️ 未找到处理后的数据，使用默认词表大小进行评估")

    # --- 2. 模型初始化 (自动加载 config 中的延迟参数) ---
    print(f"🏗️  正在构建模型 (金字塔调度: {CFG['model'].get('use_pyramid', False)})")
    model = OneTransModel(vocab_size, seq_len, CFG['model']).to(device)
    model.eval()

    # 准备输入数据 (BatchSize=1 为延迟评测标准)
    dummy_input = torch.randint(0, vocab_size, (1, seq_len)).to(device)

    # 推理精度配置 (适配 A10 BF16)
    use_amp = (device.type == 'cuda' and CFG['env'].get('use_bf16', False))
    dtype = torch.bfloat16 if use_amp else torch.float32
    
    # --- 3. 核心测量逻辑 ---
    warmups = 20
    repeats = 100
    
    print(f"🔥 热身中 (Warmup x{warmups})...")
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=dtype):
            for _ in range(warmups):
                _ = model(dummy_input)
    
    print(f"⏱️  正式测量中 (Repeats x{repeats})...")
    latencies = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=dtype):
            for i in range(repeats):
                if device.type == 'cuda':
                    torch.cuda.synchronize() # 确保 GPU 计算完成
                
                start_time = time.perf_counter()
                
                _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)

    # --- 4. 统计结果输出 ---
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)

    print("\n" + "📊 推理延迟评估报告 " + "="*30)
    print(f"🔹 测试配置: {'BF16' if use_amp else 'FP32'}, Pyramid={CFG['model'].get('use_pyramid')}")
    print(f"🔹 平均延迟 (Mean):  {avg_latency:.4f} ms")
    print(f"🔹 99分位延迟 (P99): {p99_latency:.4f} ms")
    print(f"🔹 每秒吞吐量 (QPS): {1000 / avg_latency:.2f} samples/s")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmark()