import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math

# --- 1. 配置加载 ---
CONFIG_PATH = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/config/config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device(CFG['env']['device'] if torch.cuda.is_available() else "cpu")
m_cfg = CFG['model']

# --- 2. 位置编码 (OneTrans 必备) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        return x + self.pe[:, :x.size(1)]

# --- 3. OneTrans 统一模型 ---
class OneTransModel(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        # 1. 统一 Embedding 层
        self.embedding = nn.Embedding(vocab_size, m_cfg['embed_dim'], padding_idx=0)
        self.pos_encoder = PositionalEncoding(m_cfg['embed_dim'], max_len=seq_len + 10)
        
        # 2. OneTrans 核心：统一 Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m_cfg['embed_dim'],
            nhead=m_cfg['n_heads'],
            dim_feedforward=m_cfg['embed_dim'] * 4,
            dropout=m_cfg['dropout'],
            batch_first=True,
            norm_first=True  # Pre-Norm 提高大模型训练稳定性
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=m_cfg['num_layers'])
        
        # 3. 预测头：接收 Logits (配合 BCEWithLogitsLoss)
        self.mlp = nn.Sequential(
            nn.Linear(m_cfg['embed_dim'], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(m_cfg['dropout']),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        x = self.embedding(x) * math.sqrt(m_cfg['embed_dim']) # [Batch, Seq_Len, Dim]
        x = self.pos_encoder(x)
        
        # Transformer 全局交互 (Unified Backbone)
        x = self.transformer(x)
        
        # 聚合策略：全局平均池化
        x = torch.mean(x, dim=1) 
        
        # 输出 Logits (不加 Sigmoid)
        return self.mlp(x).squeeze(-1)

# --- 4. 数据载入 ---
def load_data():
    path_train = os.path.join(CFG['paths']['processed_dir'], 'train_data.pt')
    path_val = os.path.join(CFG['paths']['processed_dir'], 'val_data.pt')
    
    # weights_only=True 提高安全性
    train_data = torch.load(path_train, weights_only=True)
    val_data = torch.load(path_val, weights_only=True)
    
    # 确保 Label 是 Float 且形状正确
    train_set = TensorDataset(train_data['x'], train_data['y'].float())
    val_set = TensorDataset(val_data['x'], val_data['y'].float())
    
    return train_set, val_set, train_data['x'].shape[1]

# --- 5. 训练主流程 ---
def train():
    # 1. 准备阶段
    train_set, val_set, seq_len = load_data()
    with open(CFG['paths']['mapping_path'], 'r') as f:
        # +2 考虑 PAD(0) 和 SEP(1)
        vocab_size = len(json.load(f)) + 2 
    
    train_loader = DataLoader(
        train_set, 
        batch_size=CFG['train']['batch_size'], 
        shuffle=True, 
        num_workers=8, # A10 性能强，可适当调高
        pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=CFG['train']['batch_size'], pin_memory=True)

    model = OneTransModel(vocab_size, seq_len).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CFG['train']['lr']))
    
    # 核心修改：使用 BCEWithLogitsLoss 提高稳定性
    criterion = nn.BCEWithLogitsLoss()
    
    # A10 混合精度加速
    use_amp = CFG['env']['use_bf16'] and DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print(f"🚀 OneTrans 训练启动 | 设备: {DEVICE} | 序列长度: {seq_len} | BF16: {use_amp}")
    print(f"📊 词表规模: {vocab_size} | Embedding 维度: {m_cfg['embed_dim']}")

    best_auc = 0.0
    for epoch in range(CFG['train']['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 验证阶段 ---
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp, dtype=torch.bfloat16):
                    # 验证时手动 Sigmoid
                    out = torch.sigmoid(model(x))
                preds.extend(out.cpu().float().numpy())
                labels.extend(y.numpy())
        
        auc = roc_auc_score(labels, preds)
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch} | Loss: {avg_loss:.4f} | Val AUC: {auc:.4f}")

        # 保存最优模型
        if auc > best_auc:
            best_auc = auc
            save_path = CFG['paths']['model_save_path']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 新纪录！模型已保存至 {save_path}")

if __name__ == "__main__":
    # A10 显存分配策略优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train()