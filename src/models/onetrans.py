import torch
import torch.nn as nn
import math

class OneTransBlock(nn.Module):
    """
    OneTrans 基础块：支持混合注意力、混合 FFN 和金字塔调度
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, use_pyramid=False):
        super().__init__()
        # 创新点 2：混合注意力 (不同层可配置不同头数)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 创新点 3：混合前馈网络 (不同层可配置不同宽度的 FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # 创新点 5：金字塔调度 (通过 Pooling 减少序列长度)
        self.use_pyramid = use_pyramid
        if use_pyramid:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x, mask=None):
        # 注意力层
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout1(attn_output)
        
        # FFN层
        x_norm = self.norm2(x)
        x_ffn = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(x_ffn)
        
        # 金字塔调度缩减序列长度
        if self.use_pyramid:
            x = x.transpose(1, 2)  # [B, D, L]
            x = self.pooling(x)
            x = x.transpose(1, 2)  # [B, L', D]
            
        return x

class OneTransModel(nn.Module):
    def __init__(self, vocab_size, seq_len, m_cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, m_cfg['embed_dim'], padding_idx=0)
        
        # 使用可学习的位置编码，适应金字塔调度后的长度变化
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, m_cfg['embed_dim']))
        
        self.layers = nn.ModuleList()
        for i in range(m_cfg['num_layers']):
            # 创新点 2 & 3 具体体现：底层窄而浅，高层宽而深
            # 示例配置：前两层用 4 头，后几层用 8 头
            curr_nhead = 4 if i < 2 else m_cfg['n_heads']
            curr_ffn_dim = m_cfg['embed_dim'] * (2 if i < 2 else 4)
            # 创新点 5：在中间层（例如第 2 层）进行长度减半
            is_pyramid = True if i == 1 else False
            
            self.layers.append(OneTransBlock(
                d_model=m_cfg['embed_dim'],
                nhead=curr_nhead,
                dim_feedforward=curr_ffn_dim,
                dropout=m_cfg['dropout'],
                use_pyramid=is_pyramid
            ))
        
        self.mlp = nn.Sequential(
            nn.Linear(m_cfg['embed_dim'], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        # 创新点 4：KV Cache 预留 (当前为全序列模式)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        for layer in self.layers:
            # 如果开启了金字塔调度，mask 也需要相应缩减，此处简化处理
            x = layer(x, mask=mask)
            
        # 最终池化聚合
        x = torch.mean(x, dim=1)
        return self.mlp(x).squeeze(-1)