import torch
from src.models.onetrans import OneTransModel
# from src.models.hyformer import HyFormerModel  # 预留给后续 HyFormer

def get_model(vocab_size, seq_len, m_cfg):
    """
    模型工厂：根据配置动态选择模型架构
    """
    model_type = m_cfg.get('type', 'onetrans').lower()
    
    if model_type == 'onetrans':
        return OneTransModel(vocab_size, seq_len, m_cfg)
    # elif model_type == 'hyformer':
    #     return HyFormerModel(vocab_size, seq_len, m_cfg)
    else:
        raise ValueError(f"❌ 未知的模型类型: {model_type}")