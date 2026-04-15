import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_onetrans_mask(L_ns: int, L_s: int, device: torch.device) -> torch.Tensor:
    """
    生成OneTrans统一注意力掩码
    
    Args:
        L_ns: 非序列Token数量
        L_s: 序列Token数量
        device: 设备
    
    Returns:
        mask: 形状为 (L_ns + L_s, L_ns + L_s) 的掩码矩阵
    """
    total_length = L_ns + L_s
    
    # 初始化全1掩码
    mask = torch.ones((total_length, total_length), device=device, dtype=torch.bool)
    
    # 对S-tokens区域（后L_s行）应用因果掩码
    if L_s > 0:
        # 生成下三角矩阵作为因果掩码
        causal_mask = torch.tril(torch.ones((L_s, L_s), device=device, dtype=torch.bool))
        # 将因果掩码应用到S-tokens区域
        mask[L_ns:, L_ns:] = causal_mask
    
    return mask


class OneTransAttention(nn.Module):
    """
    OneTrans统一注意力机制，支持NS-tokens和S-tokens的不同注意力模式
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        """
        初始化OneTrans注意力层
        
        Args:
            dim: 输入维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
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
            attn_mask: 自定义注意力掩码，形状为 (B, L, L) 或 (L, L)
            is_causal: 是否使用因果掩码（仅对S-tokens区域）
        
        Returns:
            输出张量，形状为 (B, L, D)
        """
        batch_size, seq_length, dim = x.shape
        
        # 确保输入维度匹配
        assert dim == self.dim, f"Input dimension mismatch: expected {self.dim}, got {dim}"
        
        L_s = seq_length - L_ns
        
        # 生成OneTrans掩码（如果需要）
        if is_causal and L_s > 0:
            # 获取设备
            device = x.device
            
            # 生成OneTrans统一掩码
            onetrans_mask = get_onetrans_mask(L_ns, L_s, device)
            
            # 如果提供了自定义掩码，则将其与OneTrans掩码合并
            if attn_mask is not None:
                # 确保自定义掩码形状正确
                if attn_mask.dim() == 2:
                    # 扩展到批次维度
                    attn_mask = attn_mask.unsqueeze(0)
                
                # 确保OneTrans掩码与自定义掩码形状匹配
                onetrans_mask_expanded = onetrans_mask.unsqueeze(0)
                
                # 合并掩码：两个掩码都为1的位置才允许注意力
                attn_mask = attn_mask & onetrans_mask_expanded
            else:
                # 使用OneTrans掩码
                attn_mask = onetrans_mask.unsqueeze(0)  # (1, L, L)
        
        # 投影查询、键、值
        q = self.q_proj(x)  # (B, L, D)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)
        
        # 重塑为多头注意力格式
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D/H)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D/H)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D/H)
        
        # 处理注意力掩码维度
        if attn_mask is not None:
            # scaled_dot_product_attention 期望掩码形状为 (B, H, L, L)
            # 我们当前的掩码形状是 (B, L, L)，需要扩展到多头维度
            if attn_mask.dim() == 3:
                # 扩展到注意力头维度
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, L, L)
        
        # 使用Flash Attention
        # scaled_dot_product_attention 自动处理多头注意力和掩码
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  # 我们已经手动处理了因果掩码
        )  # (B, H, L, D/H)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, dim)  # (B, L, D)
        
        # 输出投影
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


# 示例用法
if __name__ == "__main__":
    # 配置参数
    batch_size = 2
    L_ns = 8  # 非序列Token数量
    L_s = 2048  # 序列Token数量（模拟长序列）
    seq_length = L_ns + L_s
    dim = 128
    num_heads = 4
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_length, dim)
    
    # 测试掩码生成
    print("=== 测试掩码生成 ===")
    mask = get_onetrans_mask(L_ns, L_s, device=x.device)
    print(f"掩码形状: {mask.shape}")
    print(f"前L_ns行 (NS-tokens) 掩码示例:")
    print(mask[:2, :10])  # 显示前2行，前10列
    print(f"S-tokens区域掩码示例 (下三角):")
    print(mask[L_ns:L_ns+2, L_ns:L_ns+10])  # 显示S-tokens区域的前2行，前10列
    print()
    
    # 测试注意力层
    print("=== 测试OneTransAttention ===")
    attention = OneTransAttention(dim=dim, num_heads=num_heads)
    output = attention(x, L_ns=L_ns, is_causal=True)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print()
    
    # 测试注意力层（带自定义掩码）
    print("=== 测试OneTransAttention（带自定义掩码） ===")
    # 创建自定义掩码（随机掩码一些位置）
    custom_mask = torch.rand(batch_size, seq_length, seq_length) > 0.9
    output = attention(x, L_ns=L_ns, attn_mask=custom_mask, is_causal=True)
    print(f"输入形状: {x.shape}")
    print(f"自定义掩码形状: {custom_mask.shape}")
    print(f"输出形状: {output.shape}")
    print("完成！")