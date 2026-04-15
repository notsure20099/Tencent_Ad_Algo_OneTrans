import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (参考Llama架构)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMS (不包含偏置参数，与Llama一致)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并应用权重
        return self.weight * (x / rms)

class SwiGLU(nn.Module):
    """SwiGLU激活函数 (参考Llama架构)"""
    def __init__(self, dim_in: int, dim_out: int, dim_inner: Optional[int] = None):
        super().__init__()
        dim_inner = dim_inner if dim_inner is not None else dim_out * 2
        self.w1 = nn.Linear(dim_in, dim_inner)
        self.w2 = nn.Linear(dim_in, dim_inner)
        self.w3 = nn.Linear(dim_inner, dim_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: xW1 * σ(xW2) → W3
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.w3(x1 * F.silu(x2))

class PositionEmbedding(nn.Module):
    """可学习的绝对位置编码"""
    def __init__(self, max_seq_length: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, D)
        batch_size, seq_length, _ = x.shape
        # 创建位置索引: [0, 1, 2, ..., seq_length-1]
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        # 获取位置嵌入
        position_embeds = self.embedding(positions)
        # 添加到输入张量
        return x + position_embeds

class OneTransLinear(nn.Module):
    """OneTrans Linear层，混合使用Token-specific和Shared权重"""
    def __init__(self, dim_in: int, dim_out: int, l_ns: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.l_ns = l_ns  # 非序列Token数量
        
        # Token-specific weights: 前L_ns个Token有独立的权重
        self.token_specific_weights = nn.Parameter(torch.randn(l_ns, dim_in, dim_out))
        
        # Shared weight: 后L_s个Token共享一个权重
        self.shared_weight = nn.Parameter(torch.randn(dim_in, dim_out))
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(dim_out))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, D_in)
        batch_size, seq_length, dim_in = x.shape
        
        # 确保输入维度匹配
        assert dim_in == self.dim_in, f"Input dimension mismatch: expected {self.dim_in}, got {dim_in}"
        
        # 计算Token-specific部分（前L_ns个Token）
        if seq_length >= self.l_ns:
            # 分离前L_ns个Token和剩余部分
            x_ns = x[:, :self.l_ns, :]  # (B, L_ns, D_in)
            x_s = x[:, self.l_ns:, :]    # (B, L_s, D_in), L_s = L - L_ns
            
            # 使用torch.einsum进行批量矩阵乘法
            # (B, L_ns, D_in) × (L_ns, D_in, D_out) → (B, L_ns, D_out)
            output_ns = torch.einsum('BLD,LDO->BLO', x_ns, self.token_specific_weights)
            
            # 使用Shared weight处理剩余部分
            # (B, L_s, D_in) × (D_in, D_out) → (B, L_s, D_out)
            output_s = torch.matmul(x_s, self.shared_weight)
            
            # 拼接结果
            output = torch.cat([output_ns, output_s], dim=1)
        else:
            # 如果序列长度小于L_ns，只使用Token-specific部分
            output = torch.einsum('BLD,LDO->BLO', x, self.token_specific_weights[:seq_length])
        
        # 添加偏置
        output += self.bias
        
        return output

# 示例用法
if __name__ == "__main__":
    # 配置参数
    batch_size = 2
    seq_length = 100  # 包含L_ns个非序列Token和L_s个序列Token
    dim_in = 128
    dim_out = 256
    l_ns = 8  # 非序列Token数量
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_length, dim_in)
    
    # 测试OneTransLinear
    print("=== 测试OneTransLinear ===")
    onetrans_linear = OneTransLinear(dim_in, dim_out, l_ns)
    output = onetrans_linear(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print()
    
    # 测试RMSNorm
    print("=== 测试RMSNorm ===")
    rms_norm = RMSNorm(dim_in)
    norm_output = rms_norm(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {norm_output.shape}")
    print(f"输出均值: {norm_output.mean().item():.4f}, 方差: {norm_output.var().item():.4f}")
    print()
    
    # 测试SwiGLU
    print("=== 测试SwiGLU ===")
    swiglu = SwiGLU(dim_in, dim_out)
    swiglu_output = swiglu(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {swiglu_output.shape}")
    print()
    
    # 测试PositionEmbedding
    print("=== 测试PositionEmbedding ===")
    pos_embedding = PositionEmbedding(seq_length, dim_in)
    pos_output = pos_embedding(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {pos_output.shape}")
    print(f"第一个位置嵌入: {pos_embedding.embedding.weight[0, :5].tolist()}")
    print(f"第二个位置嵌入: {pos_embedding.embedding.weight[1, :5].tolist()}")