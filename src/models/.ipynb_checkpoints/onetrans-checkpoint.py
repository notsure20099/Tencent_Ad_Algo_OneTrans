import torch
import torch.nn as nn
import math

class OneTransBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, use_pyramid=False, stride=2):
        super().__init__()
        # 创新点 2：混合注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 创新点 3：混合 FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # 创新点 5：金字塔调度
        self.use_pyramid = use_pyramid
        if use_pyramid:
            # 采用卷积或池化进行序列压缩，stride 越大延迟越低
            self.pooling = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        # 注意：如果使用了金字塔压缩，后续层的 mask 需要重新计算，此处暂不处理 mask
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout1(attn_output)
        
        x_norm = self.norm2(x)
        x_ffn = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(x_ffn)
        
        if self.use_pyramid:
            x = x.transpose(1, 2)
            x = self.pooling(x)
            x = x.transpose(1, 2)
        return x

class OneTransModel(nn.Module):
    def __init__(self, vocab_size, seq_len, m_cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, m_cfg['embed_dim'], padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, m_cfg['embed_dim']))
        
        self.layers = nn.ModuleList()
        for i in range(m_cfg['num_layers']):
            # 混合注意力：前两层头数减半以降低延迟
            curr_nhead = m_cfg['n_heads'] // 2 if i < 2 else m_cfg['n_heads']
            curr_ffn = m_cfg['embed_dim'] * (2 if i < 2 else 4)
            
            # 从配置读取金字塔参数
            is_pyramid = (i == m_cfg.get('compression_layer', 1)) if m_cfg.get('use_pyramid') else False
            
            self.layers.append(OneTransBlock(
                d_model=m_cfg['embed_dim'],
                nhead=curr_nhead,
                dim_feedforward=curr_ffn,
                dropout=m_cfg['dropout'],
                use_pyramid=is_pyramid,
                stride=m_cfg.get('compression_stride', 2)
            ))
        
        self.mlp = nn.Sequential(
            nn.Linear(m_cfg['embed_dim'], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        # 修正：Embedding 缩放
        d_model = self.embedding.embedding_dim
        x = self.embedding(x) * math.sqrt(d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = torch.mean(x, dim=1)
        return self.mlp(x).squeeze(-1)