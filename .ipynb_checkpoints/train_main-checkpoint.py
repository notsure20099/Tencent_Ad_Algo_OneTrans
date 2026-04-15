import torch
import torch.nn as nn
import yaml
import os
import json
import gc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
# ==========================================
# 1. 配置加载与环境适配
# ==========================================
def load_config(path="/mnt/workspace/Tencent_Ad_Algo_OneTrans/config/config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 环境降级逻辑：如果 YAML 设为 cuda 但实际不可用
    if config['env']['device'] == "cuda" and not torch.cuda.is_available():
        print("⚠️ 检测到 CUDA 不可用，自动降级至 CPU 模式并缩小模型规模")
        config['env']['device'] = "cpu"
        config['model']['embed_dim'] = 16
        config['train']['batch_size'] = 4
        
    return config

CFG = load_config()
DEVICE = torch.device(CFG['env']['device'])

# ==========================================
# 2. OneTrans 模型架构
# ==========================================
class OneTransModel(nn.Module):
    def __init__(self, vocab_size, num_feats):
        super().__init__()
        m_cfg = CFG['model']
        self.embedding = nn.Embedding(vocab_size, m_cfg['embed_dim'], padding_idx=0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m_cfg['embed_dim'],
            nhead=m_cfg['n_heads'],
            dim_feedforward=m_cfg['embed_dim'] * 4,
            batch_first=True,
            dropout=m_cfg['dropout']
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=m_cfg['num_layers'])
        
        self.fc = nn.Sequential(
            nn.Linear(m_cfg['embed_dim'], 64),
            nn.ReLU(),
            nn.Dropout(m_cfg['dropout']),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, f, s = x.shape
        x = x.view(b * f, s) 
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) 
        x = x.view(b, f, -1).mean(dim=1) 
        return self.fc(x).squeeze()

# ==========================================
# 3. 功能函数
# ==========================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    use_amp = (DEVICE.type == 'cuda' and CFG['env']['use_bf16'])
    
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # 兼容旧版 autocast 的通用写法
        with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp, dtype=torch.bfloat16):
            pred = model(x)
        all_preds.extend(pred.cpu().float().numpy())
        all_labels.extend(y.cpu().float().numpy())
    
    try:
        return roc_auc_score(all_labels, all_preds)
    except:
        return 0.5

def load_data_from_pt(filename):
    path = os.path.join(CFG['paths']['processed_dir'], filename)
    if not os.path.exists(path):
        return None, 0, 0
    
    data = torch.load(path, weights_only=True)
    # 标签转换: 1 -> 0.0 (负), 2 -> 1.0 (正)
    y_target = (data['y'] == CFG['train']['positive_label']).float()
    
    dataset = TensorDataset(data['x'], y_target)
    num_f, s_len = data['x'].shape[1], data['x'].shape[2]
    
    del data
    gc.collect()
    return dataset, num_f, s_len

# ==========================================
# 4. 训练主流程
# ==========================================
def train():
    print(f"🔥 训练开始 | 设备: {DEVICE} | 混合精度: {CFG['env']['use_bf16']}")
    
    with open(CFG['paths']['mapping_path'], 'r') as f:
        vocab_size = len(json.load(f)) + 1

    # 1. 加载数据
    print("📂 正在载入数据集...")
    train_set, num_feats, _ = load_data_from_pt('train_data.pt')
    val_set, _, _ = load_data_from_pt('val_data.pt')
    
    if train_set is None:
        print("❌ 未发现 .pt 文件，请检查 data/processed 目录")
        return

    train_loader = DataLoader(
        train_set, 
        batch_size=CFG['train']['batch_size'], 
        shuffle=True,
        num_workers=4,       # A10 实例通常有 8-16 核，设为 4-8 比较合适
        pin_memory=True      # 必须开启，GPU 训练的标配加速
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=CFG['train']['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    # 2. 初始化
    model = OneTransModel(vocab_size, num_feats).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CFG['train']['lr']))
    criterion = nn.BCELoss()
    
    use_amp = (DEVICE.type == 'cuda' and CFG['env']['use_bf16'])
    scaler = torch.amp.GradScaler(enabled=use_amp) if DEVICE.type == 'cuda' else None

    print(f"✅ 模型构建完成。词表大小: {vocab_size}, 特征数: {num_feats}")
    print(f"📏 训练批次: {len(train_loader)} batches | 验证批次: {len(val_loader)} batches")
    print("-" * 50)

    best_auc = 0.0
    start_time = time.time()

    for epoch in range(CFG['train']['epochs']):
        model.train()
        epoch_loss = 0
        
        # --- 引入 tqdm 进度条 ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", unit="batch", leave=False)
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp, dtype=torch.bfloat16):
                pred = model(x)
                loss = criterion(pred, y)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # 动态更新进度条右侧的 Loss 显示
            if batch_idx % 5 == 0:
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        # --- 每个 Epoch 结束后的深度分析 ---
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"⌛ 正在验证 Epoch {epoch}...")
        current_auc = evaluate(model, val_loader)
        
        elapsed = (time.time() - start_time) / 60
        print(f"📊 [Epoch {epoch:02d}] 耗时: {elapsed:.1f}min | Avg Loss: {avg_loss:.4f} | Val AUC: {current_auc:.4f}")

        # 存档最优模型
        if current_auc > best_auc:
            best_auc = current_auc
            os.makedirs(os.path.dirname(CFG['paths']['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), CFG['paths']['model_save_path'])
            print(f"⭐ 发现更优模型! 存档中... (Best AUC: {best_auc:.4f})")
        
        print("-" * 50)

    print(f"✅ 任务全部跑通! 累计总耗时: {(time.time() - start_time)/60:.2f} min")

if __name__ == "__main__":
    train()