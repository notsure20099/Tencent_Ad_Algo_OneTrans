import os
import yaml
import torch
import sys
from torch.utils.data import DataLoader, TensorDataset
from src.models.factory import get_model
from src.utils.trainer import Trainer

def auto_configure(cfg):
    """
    根据硬件环境自动调整超参数，防止 CPU 内存溢出
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"💡 检测到 GPU ({torch.cuda.get_device_name(0)}), 使用完全体配置。")
    else:
        device = torch.device("cpu")
        print("⚠️ 警告: 未检测到 GPU，切换至 [CPU 调试模式]：强制降低维度以防 Terminated。")
        
        # 强制覆盖内存敏感参数
        cfg['model']['embed_dim'] = 16        # 极大降低 Embedding 层内存占用
        cfg['model']['num_layers'] = 2       # 减少层数
        cfg['model']['n_heads'] = 2          # 减少头数
        cfg['train']['batch_size'] = 2       # CPU 模式极小 Batch
        cfg['env']['use_bf16'] = False       # CPU 暂不开启 BF16

    return device, cfg

def run_experiment():
    # 1. 加载基础配置
    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ 错误: 找不到配置文件 {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # 2. 硬件感知与参数自动覆盖
    device, cfg = auto_configure(base_cfg)

    if device.type == 'cuda':
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cudnn.benchmark = True

    # 3. 数据加载
    processed_dir = cfg['paths']['processed_dir']
    train_path = os.path.join(processed_dir, 'train_data.pt')
    val_path = os.path.join(processed_dir, 'val_data.pt')

    if not os.path.exists(train_path):
        print(f"❌ 错误: 找不到预处理数据 {train_path}，请先运行数据准备脚本。")
        return

    train_data = torch.load(train_path, weights_only=True)
    val_data = torch.load(val_path, weights_only=True)
    
    vocab_size = max(train_data['x'].max().item(), val_data['x'].max().item()) + 1
    seq_len = train_data['x'].shape[1]
    
    train_loader = DataLoader(
        TensorDataset(train_data['x'], train_data['y']), 
        batch_size=cfg['train']['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data['x'], val_data['y']), 
        batch_size=cfg['train']['batch_size'], shuffle=False
    )
    
    # 4. 初始化模型 (通过工厂)
    print(f"🚀 启动 | 模型: {cfg['model'].get('type', 'onetrans')} | 词表: {vocab_size} | 序列: {seq_len}")
    model = get_model(vocab_size, seq_len, cfg['model']).to(device)
    
    # 5. 训练准备
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    use_amp = (device.type == 'cuda' and cfg['env'].get('use_bf16', False))
    
    trainer = Trainer(model, optimizer, criterion, device, use_amp)

    # 6. 训练循环
    for epoch in range(cfg['train']['epochs']):
        train_loss = trainer.train_one_epoch(train_loader, epoch, cfg['train']['epochs'])
        val_loss, val_auc = trainer.evaluate(val_loader)
        
        print(f"\n✨ Epoch {epoch+1} 总结:")
        print(f"   [Train] Loss: {train_loss:.4f}")
        print(f"   [Val]   Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    run_experiment()