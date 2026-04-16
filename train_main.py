import os
import yaml
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from src.models.onetrans import OneTransModel

# --- 全局环境配置 ---
CONFIG_PATH = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/config/config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

# 自动检测设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. 动态对齐词表（核心：防止 IndexError）
    path_train = os.path.join(CFG['paths']['processed_dir'], 'train_data.pt')
    train_data = torch.load(path_train, weights_only=True)
    
    max_id = train_data['x'].max().item()
    vocab_size = max_id + 1
    seq_len = train_data['x'].shape[1]
    
    print(f"🛠️  初始化模型: 设备={DEVICE}, 词表大小={vocab_size}, 序列长度={seq_len}")
    
    model = OneTransModel(vocab_size, seq_len, CFG['model']).to(DEVICE)
    
    # 2. 准备数据
    train_set = TensorDataset(train_data['x'], train_data['y'])
    loader = DataLoader(train_set, batch_size=CFG['train']['batch_size'], shuffle=True)
    
    # 3. 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CFG['train']['lr']))
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 混合精度适配
    use_amp = (DEVICE.type == 'cuda' and CFG['env']['use_bf16'])
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(CFG['train']['epochs']):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            # 自动精度切换
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"Epoch {epoch} finished.")

if __name__ == "__main__":
    # GPU 性能优化参数
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    train()