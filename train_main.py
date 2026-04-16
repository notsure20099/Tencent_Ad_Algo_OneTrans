import os
import yaml
import torch
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.models.onetrans import OneTransModel

# --- 1. 环境配置 ---
CONFIG_PATH = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/config/config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, device, use_amp):
    """验证集评估函数"""
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            
            total_loss += loss.item()
            all_labels.extend(y.float().cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).float().cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc

def train():
    # --- 2. 数据加载与动态词表对齐 ---
    processed_dir = CFG['paths']['processed_dir']
    train_data = torch.load(os.path.join(processed_dir, 'train_data.pt'), weights_only=True)
    val_data = torch.load(os.path.join(processed_dir, 'val_data.pt'), weights_only=True)
    
    # 动态确定词表大小（覆盖所有已见ID）
    vocab_size = max(train_data['x'].max().item(), val_data['x'].max().item()) + 1
    seq_len = train_data['x'].shape[1]
    
    train_set = TensorDataset(train_data['x'], train_data['y'])
    val_set = TensorDataset(val_data['x'], val_data['y'])
    
    train_loader = DataLoader(train_set, batch_size=CFG['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=CFG['train']['batch_size'], shuffle=False)
    
    # --- 3. 模型与优化器初始化 ---
    print(f"🚀 启动训练 | 设备: {DEVICE} | 词表: {vocab_size} | 序列: {seq_len}")
    model = OneTransModel(vocab_size, seq_len, CFG['model']).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CFG['train']['lr']), weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    use_amp = (DEVICE.type == 'cuda' and CFG['env'].get('use_bf16', False))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # --- 4. 训练循环 ---
    for epoch in range(CFG['train']['epochs']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['train']['epochs']}")
        
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        # --- 5. 每轮结束进行验证 ---
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_auc = evaluate(model, val_loader, DEVICE, use_amp)
        
        print(f"\n✨ Epoch {epoch+1} 总结:")
        print(f"   [Train] Loss: {avg_train_loss:.4f}")
        print(f"   [Val]   Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")
        print("-" * 30)

        # 保存最优模型
        # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cudnn.benchmark = True
    train()