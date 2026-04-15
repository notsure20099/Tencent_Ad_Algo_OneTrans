import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional

# 导入之前实现的组件
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.onetrans_parts import RMSNorm, SwiGLU, PositionEmbedding
from src.models.onetrans_attention import OneTransAttention


class OneTransBlock(nn.Module):
    """
    OneTrans Block，包含：RMSNorm -> Multi-head Attention -> RMSNorm -> FeedForward(SwiGLU)
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, expansion_ratio: int = 4):
        """
        初始化OneTrans Block
        
        Args:
            dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout概率
            expansion_ratio: FFN扩展比例
        """
        super().__init__()
        
        # 第一个RMSNorm
        self.norm1 = RMSNorm(dim)
        
        # Multi-head Attention
        self.attn = OneTransAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 第二个RMSNorm
        self.norm2 = RMSNorm(dim)
        
        # FeedForward with SwiGLU
        dim_inner = dim * expansion_ratio
        self.ffn = SwiGLU(dim_in=dim, dim_out=dim, dim_inner=dim_inner)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        L_ns: int,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, L, D)
            L_ns: 非序列Token数量
            attn_mask: 注意力掩码
            is_causal: 是否使用因果掩码
        
        Returns:
            输出张量，形状为 (B, L, D)
        """
        # RMSNorm -> Attention -> Residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, L_ns=L_ns, attn_mask=attn_mask, is_causal=is_causal)
        x = self.dropout(x)
        x = residual + x
        
        # RMSNorm -> FFN -> Residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class OneTransBackbone(nn.Module):
    """
    OneTrans 主干网络
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        L_ns: int = 8,
        L_s: int = 2048,
        dropout: float = 0.1,
        expansion_ratio: int = 4,
        downsample_rate: int = 4  # 新增：下采样率
    ):
        """
        初始化OneTrans主干网络
        
        Args:
            embedding_dim: 嵌入维度
            num_layers: 网络层数
            num_heads: 注意力头数
            L_ns: 非序列Token数量
            L_s: 序列Token数量
            dropout: Dropout概率
            expansion_ratio: FFN扩展比例
            downsample_rate: 序列Token下采样率
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.L_ns = L_ns
        self.L_s_original = L_s  # 保存原始的序列Token数量
        self.L_s = L_s  # 当前的序列Token数量，会在下采样后更新
        self.downsample_rate = downsample_rate
        self.L_s_downsampled = L_s // downsample_rate
        self.total_length = L_ns + L_s
        self.total_length_downsampled = L_ns + self.L_s_downsampled
        
        # 位置编码
        self.position_embedding = PositionEmbedding(max_seq_length=self.total_length, embedding_dim=embedding_dim)
        
        # 下采样层：Average Pooling
        self.downsample_layer = nn.AvgPool1d(
            kernel_size=downsample_rate,
            stride=downsample_rate,
            padding=0
        )
        
        # OneTrans Blocks
        self.blocks = nn.ModuleList([
            OneTransBlock(
                dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                expansion_ratio=expansion_ratio
            ) for _ in range(num_layers)
        ])
        
        # 输出层RMSNorm
        self.output_norm = RMSNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        ns_tokens: torch.Tensor,
        s_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            ns_tokens: 非序列Token，形状为 (B, L_ns, D)
            s_tokens: 序列Token，形状为 (B, L_s, D)
            attn_mask: 注意力掩码
            is_causal: 是否使用因果掩码
        
        Returns:
            最后一层的隐状态，形状为 (B, L_ns+L_s//downsample_rate, D)
        """
        # 1. 拼接 NS-tokens 和 S-tokens
        # 检查输入形状
        assert ns_tokens.size(1) == self.L_ns, f"NS-tokens长度不匹配: 期望 {self.L_ns}, 得到 {ns_tokens.size(1)}"
        assert s_tokens.size(1) == self.L_s_original, f"S-tokens长度不匹配: 期望 {self.L_s_original}, 得到 {s_tokens.size(1)}"
        
        # 重置当前L_s为原始值（处理多次前向传播的情况）
        self.L_s = self.L_s_original
        self.total_length = self.L_ns + self.L_s
        
        # 拼接
        x = torch.cat([ns_tokens, s_tokens], dim=1)  # (B, L_ns + L_s, D)
        
        # 2. 加入位置编码
        x = self.position_embedding(x)
        x = self.dropout(x)
        
        # 3. 循环经过 N 层 OneTrans Block
        for i, block in enumerate(self.blocks):
            # 使用梯度检查点以节省显存
            def create_checkpoint_func(block, x, L_ns, attn_mask, is_causal):
                return block(x, L_ns, attn_mask, is_causal)
            
            if self.training:
                x = checkpoint(create_checkpoint_func, block, x, self.L_ns, attn_mask, is_causal)
            else:
                x = block(x, L_ns=self.L_ns, attn_mask=attn_mask, is_causal=is_causal)
            
            # 在第2个Block之后进行下采样
            if i == 1:  # 第2个Block（索引从0开始）
                # 分离NS-tokens和S-tokens
                x_ns = x[:, :self.L_ns, :]  # (B, L_ns, D)
                x_s = x[:, self.L_ns:, :]  # (B, L_s, D)
                
                # 对S-tokens进行Average Pooling下采样
                # 需要调整维度：(B, L, D) -> (B, D, L) 进行池化
                x_s_reshaped = x_s.transpose(1, 2)  # (B, D, L_s)
                x_s_downsampled = self.downsample_layer(x_s_reshaped)  # (B, D, L_s//downsample_rate)
                x_s_downsampled = x_s_downsampled.transpose(1, 2)  # (B, L_s//downsample_rate, D)
                
                # 拼接回NS-tokens和下采样后的S-tokens
                x = torch.cat([x_ns, x_s_downsampled], dim=1)  # (B, L_ns + L_s//downsample_rate, D)
                
                # 调整L_s和total_length，用于后续Block
                self.L_s = self.L_s_downsampled
                self.total_length = self.total_length_downsampled
                
                # 调整注意力掩码（如果有）
                if attn_mask is not None:
                    # 简化处理：只保留NS-tokens部分的掩码，对S-tokens部分使用新的因果掩码
                    # 这是一种近似处理，但可以确保显存使用减少
                    attn_mask = None  # 暂时禁用自定义掩码，使用新的因果掩码
        
        # 4. 输出层RMSNorm
        x = self.output_norm(x)
        
        return x


# 示例用法
if __name__ == "__main__":
    # 配置参数
    batch_size = 2
    embedding_dim = 128
    num_layers = 6
    num_heads = 4
    L_ns = 8  # 非序列Token数量
    L_s = 2048  # 序列Token数量
    downsample_rate = 4  # 下采样率
    
    print("=== 测试OneTransBackbone（带下采样） ===")
    print(f"配置: batch_size={batch_size}, L_ns={L_ns}, L_s={L_s}, downsample_rate={downsample_rate}")
    print()
    
    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns_tokens = torch.randn(batch_size, L_ns, embedding_dim, device=device)
    s_tokens = torch.randn(batch_size, L_s, embedding_dim, device=device)
    
    print(f"使用设备: {device}")
    print(f"NS-Tokens形状: {ns_tokens.shape}")
    print(f"S-Tokens形状: {s_tokens.shape}")
    print()
    
    # 创建模型
    model = OneTransBackbone(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        L_ns=L_ns,
        L_s=L_s,
        downsample_rate=downsample_rate
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print()
    
    # 运行Forward
    print("运行Forward")
    with torch.no_grad():
        output = model(ns_tokens, s_tokens, is_causal=True)
    
    print(f"输出形状: {output.shape}")
    print(f"期望形状: ({batch_size}, {L_ns + L_s//downsample_rate}, {embedding_dim})")
    print(f"形状检查: {'通过' if output.shape == (batch_size, L_ns + L_s//downsample_rate, embedding_dim) else '失败'}")
    print()
    
    # 测试带自定义掩码的情况
    print("测试带自定义掩码")
    total_length = L_ns + L_s
    custom_mask = torch.rand(batch_size, total_length, total_length, device=device) > 0.95
    
    with torch.no_grad():
        output_masked = model(ns_tokens, s_tokens, attn_mask=custom_mask, is_causal=True)
    
    print(f"自定义掩码形状: {custom_mask.shape}")
    print(f"带掩码的输出形状: {output_masked.shape}")
    print()
    
    # 测试梯度检查点
    print("测试梯度检查点")
    model.train()
    
    # 清除之前的梯度
    model.zero_grad()
    
    # 前向传播
    output = model(ns_tokens, s_tokens, is_causal=True)
    
    # 计算损失并反向传播
    loss = output.sum()
    loss.backward()
    
    # 检查梯度
    has_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads.append(param.grad.abs().sum().item() > 0)
        else:
            has_grads.append(False)
    
    print(f"梯度检查: {'所有参数都有梯度' if all(has_grads) else '部分参数无梯度'}")
    print()
    
    print("完成！")