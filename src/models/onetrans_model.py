import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 添加项目根目录到Python路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.onetrans_backbone import OneTransBackbone
from src.models.onetrans_parts import RMSNorm


class OneTransModel(nn.Module):
    """
    完整的OneTrans模型，包含Backbone和Prediction Head
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
        downsample_rate: int = 4,
        mlp_hidden_dims: Optional[list] = None
    ):
        """
        初始化OneTrans模型
        
        Args:
            embedding_dim: 嵌入维度
            num_layers: 网络层数
            num_heads: 注意力头数
            L_ns: 非序列Token数量
            L_s: 序列Token数量
            dropout: Dropout概率
            expansion_ratio: FFN扩展比例
            downsample_rate: 序列Token下采样率
            mlp_hidden_dims: 预测头MLP的隐藏层维度列表
        """
        super().__init__()
        
        # 配置参数
        self.embedding_dim = embedding_dim
        self.L_ns = L_ns
        self.L_s = L_s
        
        # Backbone（使用优化后的版本，支持下采样）
        self.backbone = OneTransBackbone(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            L_ns=L_ns,
            L_s=L_s,
            dropout=dropout,
            expansion_ratio=expansion_ratio,
            downsample_rate=downsample_rate
        )
        
        # Prediction Head
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128]
        
        # 3层MLP
        mlp_input_dim = L_ns * embedding_dim  # Flatten后的维度
        mlp_layers = []
        current_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # 输出层
        mlp_layers.append(nn.Linear(current_dim, 1))
        
        self.prediction_head = nn.Sequential(*mlp_layers)
    
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
            logits: 预测的CTR值，形状为 (B, 1)
        """
        # 1. 通过Backbone
        backbone_output = self.backbone(ns_tokens, s_tokens, attn_mask=attn_mask, is_causal=is_causal)
        
        # 2. 提取L_ns个Token的表示
        ns_representations = backbone_output[:, :self.L_ns, :]  # (B, L_ns, D)
        
        # 3. Flatten
        ns_flattened = ns_representations.flatten(start_dim=1)  # (B, L_ns * D)
        
        # 4. 3层MLP输出logits
        logits = self.prediction_head(ns_flattened)  # (B, 1)
        
        return logits


# 测试脚本
if __name__ == "__main__":
    # 配置参数
    batch_size = 2  # 使用较小的batch_size避免显存问题
    embedding_dim = 128
    num_layers = 6
    num_heads = 4
    L_ns = 8
    L_s = 2048
    downsample_rate = 4  # 使用下采样减少显存占用

    print("=== OneTrans模型测试 ===")
    print(f"配置: batch_size={batch_size}, L_ns={L_ns}, L_s={L_s}, downsample_rate={downsample_rate}")
    print()

    # 1. 创建测试数据
    print("1. 创建测试数据")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")

    # 模拟输入：NS-Tokens (2, 8, 128)，S-Tokens (2, 2048, 128)
    ns_tokens = torch.randn(batch_size, L_ns, embedding_dim, device=device)
    s_tokens = torch.randn(batch_size, L_s, embedding_dim, device=device)

    # 模拟标签
    labels = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)

    print(f"   NS-Tokens形状: {ns_tokens.shape}")
    print(f"   S-Tokens形状: {s_tokens.shape}")
    print(f"   标签形状: {labels.shape}")
    print()

    # 2. 创建模型
    print("2. 创建模型")
    model = OneTransModel(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        L_ns=L_ns,
        L_s=L_s,
        downsample_rate=downsample_rate
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   模型总参数量: {total_params / 1e6:.2f}M")
    print()

    # 3. 运行Forward
    print("3. 运行Forward")
    with torch.no_grad():
        logits = model(ns_tokens, s_tokens)

    print(f"   输出形状: {logits.shape}")
    print(f"   期望形状: ({batch_size}, 1)")
    print(f"   形状检查: {'通过' if logits.shape == (batch_size, 1) else '失败'}")
    print()

    # 4. 运行Backward
    print("4. 运行Backward")

    # 清除之前的梯度
    model.zero_grad()

    # 重新运行前向传播（带梯度计算）
    logits = model(ns_tokens, s_tokens)

    # 计算损失
    loss_fn = F.binary_cross_entropy_with_logits
    loss = loss_fn(logits, labels)

    print(f"   损失值: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    # 检查梯度更新
    has_grads = []
    param_names = []

    # 检查Backbone中的OneTransLinear层的梯度
    for name, param in model.named_parameters():
        if "token_specific" in name or "shared_weight" in name:
            if param.grad is not None:
                has_grads.append(param.grad.abs().sum().item() > 0)
                param_names.append(name)

    print(f"   参数梯度检查: {'全部有梯度' if all(has_grads) else '部分无梯度'}")
    for name, has_grad in zip(param_names, has_grads):
        print(f"     {name}: {'有梯度' if has_grad else '无梯度'}")
    print()

    # 5. 显存监控
    print("5. 显存监控")
    if device.type == "cuda":
        # 打印当前显存使用情况
        current_memory = torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024  # GB
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024  # GB
        
        print(f"   当前显存使用: {current_memory:.2f} GB")
        print(f"   峰值显存使用: {peak_memory:.2f} GB")
    else:
        print("   使用CPU，无法监控显存")

    print("\n=== 测试完成 ===")