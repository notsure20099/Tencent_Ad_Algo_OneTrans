import os
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.models.onetrans import OneTransModel # 引用新模型
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --- 配置与设备 ---
CONFIG_PATH = '/mnt/workspace/Tencent_Ad_Algo_OneTrans/config/config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device(CFG['env']['device'] if torch.cuda.is_available() else "cpu")

def train():
    # 加载数据逻辑 (保持不变)
    path_train = os.path.join(CFG['paths']['processed_dir'], 'train_data.pt')
    train_data = torch.load(path_train, weights_only=True)
    train_set = TensorDataset(train_data['x'], train_data['y'].float())
    seq_len = train_data['x'].shape[1]
    
    # 实例化重构后的 OneTrans 模型
    max_id_in_data = train_data['x'].max().item()
    vocab_size = max_id_in_data + 1
    print(f"检测到数据中最大 ID 为: {max_id_in_data}, 设置 Vocab Size 为: {vocab_size}")
    model = OneTransModel(vocab_size, seq_len, CFG['model']).to(DEVICE)
    
    # 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CFG['train']['lr']), weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # A10 混合精度
    use_amp = CFG['env']['use_bf16']
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(CFG['train']['epochs']):
        model.train()
        for x, y in tqdm(DataLoader(train_set, batch_size=CFG['train']['batch_size'], shuffle=True)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 验证逻辑省略... (调用 model.eval())

if __name__ == "__main__":
    train()