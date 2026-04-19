import torch
import time
import numpy as np
import yaml
import os
from src.models.factory import get_model

def run_benchmark():
    CONFIG_PATH = './config/config.yaml'
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 同步硬件感知逻辑，确保 benchmark 环境与训练环境一致
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        cfg['model']['embed_dim'] = 16 
        cfg['model']['num_layers'] = 2
        cfg['model']['n_heads'] = 2

    # 获取数据维度用于模型初始化
    path_train = os.path.join(cfg['paths']['processed_dir'], 'train_data.pt')
    if os.path.exists(path_train):
        data = torch.load(path_train, weights_only=True)
        vocab_size = data['x'].max().item() + 1
        seq_len = data['x'].shape[1]
    else:
        vocab_size, seq_len = 2005000, 1507

    # 使用工厂获取模型
    model = get_model(vocab_size, seq_len, cfg['model']).to(device)
    model.eval()

    # BatchSize=1 是官方延迟评测标准
    dummy_input = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    use_amp = (device.type == 'cuda' and cfg['env'].get('use_bf16', False))
    dtype = torch.bfloat16 if use_amp else torch.float32
    
    print(f"\n⏱️  延迟评估启动 | 设备: {device} | 模型: {cfg['model'].get('type')}")
    print(f"🔹 维度: {cfg['model']['embed_dim']}, 层数: {cfg['model']['num_layers']}")
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=dtype):
            # 热身
            for _ in range(20): _ = model(dummy_input)
            
            # 测量
            latencies = []
            for _ in range(100):
                if device.type == 'cuda': torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(dummy_input)
                if device.type == 'cuda': torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

    print("-" * 40)
    print(f"📊 平均延迟 (Mean): {np.mean(latencies):.4f} ms")
    print(f"📊 99分位延迟 (P99): {np.percentile(latencies, 99):.4f} ms")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    run_benchmark()