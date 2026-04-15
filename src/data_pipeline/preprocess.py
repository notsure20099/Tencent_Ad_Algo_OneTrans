import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union, Optional

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化
        norm_x = x / rms
        # 应用权重和偏置
        return self.weight * norm_x + self.bias

class OneTransPreprocessor:
    def __init__(self, config_path: str):
        """初始化OneTrans预处理类"""
        self.config_path = config_path
        self.feature_config = self._load_config()
        
        # 特征分组
        # 功能一：NS-Tokenizer特征
        self.id_features = ['user_id', 'item_id']
        self.user_int_scalar_features = self.feature_config['user_int_features']['scalar_columns']
        self.item_int_scalar_features = self.feature_config['item_int_features']['scalar_columns']
        self.all_scalar_features = self.id_features + self.user_int_scalar_features + self.item_int_scalar_features
        
        # 功能二：Dense-to-Token Mapper特征
        self.user_int_array_features = self.feature_config['user_int_features']['array_columns']
        self.user_dense_array_features = self.feature_config['user_dense_features']['array_columns']
        self.item_int_array_features = self.feature_config['item_int_features']['array_columns']
        
        # 默认配置
        # 功能一配置
        self.hash_bucket_size = 1000000  # ID特征的哈希桶大小
        self.num_buckets = 128  # 数值特征的分桶数量
        self.l_ns = 8  # 非序列特征的Token数量
        
        # 功能二配置
        self.transformer_hidden_dim = 128  # Transformer隐层维度
        
        # 初始化功能一组件
        self.projection_layers = self._init_projection_layers()
        self.rms_norm = RMSNorm(self.l_ns)
        
        # 初始化功能二组件
        self.shared_mlp = self._init_shared_mlp()
        self.fid_feature_mapping = self._group_features_by_fid()
        # 初始化RMSNorm层 (功能二)
        self.dense_token_rms_norm = RMSNorm(self.transformer_hidden_dim)
        
        # 功能三：S-Tokenizer配置
        self.max_seq_length = 2048  # 最大序列长度
        self.vocab_size = 1000000  # 词汇表大小
        self.embedding_dim = self.transformer_hidden_dim  # 嵌入维度与Transformer隐层维度一致
        self.sep_token_id = 0  # [SEP] Token的ID
        
        # 初始化功能三组件
        self.shared_embedding = self._init_shared_embedding()
        self.domain_sequence_features = self.feature_config['domain_sequence_features']
    
    def _group_features_by_fid(self) -> Dict[str, List[str]]:
        """将特征按fid分组（相同数字后缀的特征）"""
        import re
        
        # 提取所有数组特征
        all_array_features = (self.user_int_array_features + 
                             self.user_dense_array_features + 
                             self.item_int_array_features)
        
        # 按fid分组
        fid_groups = {}
        
        for feat in all_array_features:
            # 提取特征名中的数字部分作为fid
            match = re.search(r'_feats_(\d+)', feat)
            if match:
                fid = match.group(1)
                if fid not in fid_groups:
                    fid_groups[fid] = []
                fid_groups[fid].append(feat)
        
        print(f"\n=== 按fid分组数组特征 ===")
        for fid, feats in fid_groups.items():
            print(f"  FID {fid}: {feats}")
        print(f"共 {len(fid_groups)} 个fid组")
        
        return fid_groups
    
    def _init_shared_mlp(self) -> nn.Sequential:
        """初始化用于Dense-to-Token映射的Shared MLP"""
        # 这里假设输入特征的最大维度，实际使用时会根据输入动态调整
        # 或者使用自适应的网络结构
        mlp = nn.Sequential(
            nn.Linear(1, self.transformer_hidden_dim),  # 输入维度会在forward时动态调整
            nn.ReLU(),
            nn.Linear(self.transformer_hidden_dim, self.transformer_hidden_dim)
        )
        
        print(f"\n=== 初始化Shared MLP ===")
        print(f"MLP结构: {mlp}")
        print(f"Transformer隐层维度d: {self.transformer_hidden_dim}")
        
        return mlp
    
    def _init_shared_embedding(self) -> nn.Embedding:
        """初始化用于序列特征的共享嵌入层"""
        # 创建共享的Embedding层
        embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0  # 使用0作为填充标记
        )
        
        print(f"\n=== 初始化Shared Embedding ===")
        print(f"Embedding维度: {self.embedding_dim}")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"填充标记: {0}")
        print(f"SEP Token ID: {self.sep_token_id}")
        
        return embedding
    
    def _process_array_feature(self, df: pd.DataFrame, col: str) -> np.ndarray:
        """处理单个数组特征，将其转换为统一长度的数组"""
        # 获取数组的最大长度
        max_len = df[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max()
        print(f"  特征 {col}: 最大长度 {max_len}")
        
        # 填充或截断到最大长度
        processed = []
        for arr in df[col]:
            if isinstance(arr, (list, np.ndarray)):
                # 截断过长的数组
                if len(arr) > max_len:
                    arr = arr[:max_len]
                # 填充过短的数组
                elif len(arr) < max_len:
                    # 根据特征类型选择填充值
                    if col.startswith('user_dense_'):
                        arr = np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0.0)
                    else:
                        arr = np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0)
                processed.append(arr)
            else:
                # 处理空值
                if col.startswith('user_dense_'):
                    processed.append(np.zeros(max_len, dtype=np.float32))
                else:
                    processed.append(np.zeros(max_len, dtype=np.int64))
        
        return np.array(processed)
    
    def _load_config(self) -> Dict:
        """加载特征配置文件"""
        import json
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _init_projection_layers(self) -> nn.ModuleList:
        """初始化每个Token的独立投影参数"""
        # 计算输入特征的总维度
        input_dim = len(self.all_scalar_features)
        print(f"NS-Tokenizer投影层初始化: 输入维度={input_dim}, 输出Token数={self.l_ns}")
        
        # 为每个Token创建独立的投影层
        projection_layers = nn.ModuleList()
        for i in range(self.l_ns):
            # 每个投影层将输入映射到1维
            projection_layers.append(nn.Linear(input_dim, 1))
            print(f"  Token {i+1}投影层: Linear({input_dim}, 1)")
        
        return projection_layers
    
    def _hash_bucketize(self, value: Union[int, str], bucket_size: int) -> int:
        """对ID特征进行Hash分桶"""
        import hashlib
        if isinstance(value, str):
            value_str = value
        else:
            value_str = str(value)
        
        # 使用SHA256哈希
        hash_val = hashlib.sha256(value_str.encode()).hexdigest()
        # 转换为整数并取模得到桶索引
        bucket_idx = int(hash_val, 16) % bucket_size
        
        return bucket_idx
    
    def _numeric_bucketize(self, value: float, num_buckets: int, min_val: float, max_val: float) -> int:
        """对数值特征进行分桶，处理NaN值"""
        # 处理NaN值，将其分配到第0桶
        if pd.isna(value):
            return 0
        
        # 处理边界情况
        if value <= min_val:
            return 0
        if value >= max_val:
            return num_buckets - 1
        
        # 线性映射到桶索引
        bucket_idx = int((value - min_val) / (max_val - min_val) * num_buckets)
        # 确保不超过最大桶索引
        bucket_idx = min(bucket_idx, num_buckets - 1)
        
        return bucket_idx
    
    def fit(self, df: pd.DataFrame) -> "OneTransPreprocessor":
        """拟合预处理器，计算数值特征的统计信息"""
        print("=== 拟合OneTrans预处理器 ===")
        
        # 计算数值特征的min和max用于分桶
        self.numeric_stats = {}
        for col in self.all_scalar_features:
            if df[col].dtype in ['int64', 'float64']:
                # 忽略NaN值计算min和max
                self.numeric_stats[col] = {
                    'min': df[col].min(skipna=True),
                    'max': df[col].max(skipna=True)
                }
                print(f"  特征 {col}: min={self.numeric_stats[col]['min']:.4f}, max={self.numeric_stats[col]['max']:.4f}, NaN数量={df[col].isna().sum()}")
        
        print(f"\n共处理 {len(self.numeric_stats)} 个数值特征")
        return self
    
    def transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """转换数据，生成NS-Tokens"""
        print("\n=== 数据转换开始 ===")
        
        # 1. 处理所有标量特征
        processed_features = []
        print(f"\n=== NS-Tokenizer 处理开始 ===")
        print(f"待处理的标量特征数量: {len(self.all_scalar_features)}")
        print(f"特征列表: {self.all_scalar_features[:5]}..." if len(self.all_scalar_features) > 5 else f"特征列表: {self.all_scalar_features}")
        
        for i, col in enumerate(self.all_scalar_features):
            if col in self.id_features:
                # ID特征：Hash分桶
                bucketized = df[col].apply(lambda x: self._hash_bucketize(x, self.hash_bucket_size)).values
                print(f"  [{i+1}/{len(self.all_scalar_features)}] ID特征 {col}: 处理完成，样本数 {len(bucketized)}")
            else:
                # 数值特征：Bucketization
                stats = self.numeric_stats[col]
                bucketized = df[col].apply(
                    lambda x: self._numeric_bucketize(x, self.num_buckets, stats['min'], stats['max'])
                ).values
                print(f"  [{i+1}/{len(self.all_scalar_features)}] 数值特征 {col}: 处理完成，范围 [{stats['min']:.4f}, {stats['max']:.4f}], 样本数 {len(bucketized)}")
            
            processed_features.append(bucketized)
        
        # 将所有特征拼接成一个矩阵 (batch_size, num_features)
        features_matrix = np.stack(processed_features, axis=1)
        print(f"\n特征矩阵形状: {features_matrix.shape} (batch_size, num_features)")
        print(f"特征矩阵示例 (前2行前3列):\n{features_matrix[:2, :3]}")
        
        # 转换为PyTorch张量
        features_tensor = torch.tensor(features_matrix, dtype=torch.float32)
        print(f"PyTorch张量形状: {features_tensor.shape}")
        print(f"张量数据类型: {features_tensor.dtype}")
        print(f"张量均值: {features_tensor.mean().item():.4f}, 方差: {features_tensor.var().item():.4f}")
        
        # 2. 应用NS-Tokenizer (Auto-Split Tokenizer)
        print(f"\n=== 应用Auto-Split Tokenizer ===")
        print(f"投影层数 (L_NS): {self.l_ns}")
        
        ns_tokens = []
        for i, projection in enumerate(self.projection_layers):
            # 每个投影层生成一个Token
            token = projection(features_tensor)
            ns_tokens.append(token)
            print(f"  Token {i+1}: 形状 {token.shape}, 均值 {token.mean().item():.6f}, 方差 {token.var().item():.6f}")
            print(f"    权重形状: {projection.weight.shape}, 偏置形状: {projection.bias.shape}")
        
        # 将所有Tokens拼接 (batch_size, l_ns)
        ns_tokens = torch.cat(ns_tokens, dim=1)
        print(f"\n所有NS-Tokens拼接后形状: {ns_tokens.shape}")
        print(f"NS-Tokens示例 (前2行):\n{ns_tokens[:2]}")
        print(f"NS-Tokens均值: {ns_tokens.mean().item():.4f}, 方差: {ns_tokens.var().item():.4f}")
        
        # 应用RMSNorm层 (Pre-norm设计)
        print(f"\n=== 应用RMSNorm归一化 ===")
        print(f"RMSNorm输入形状: {ns_tokens.shape}")
        ns_tokens_norm = self.rms_norm(ns_tokens)
        print(f"RMSNorm输出形状: {ns_tokens_norm.shape}")
        print(f"RMSNorm输出均值: {ns_tokens_norm.mean().item():.4f}, 方差: {ns_tokens_norm.var().item():.4f}")
        print(f"RMSNorm输出示例 (前2行):\n{ns_tokens_norm[:2]}")
        
        print(f"=== NS-Tokenizer 处理完成 ===")
        
        # 3. 返回处理结果
        result = {
            'ns_tokens': ns_tokens_norm,
            'ns_tokens_raw': ns_tokens,  # 保留原始未归一化的Tokens用于比较
            'scalar_features': features_tensor
        }
        
        print("\n=== 数据转换完成 ===")
        return result
    
    def transform_dense_token(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """功能二：Dense-to-Token Mapper (Dense & Int Array Processing)
        
        输入：user_dense_feats (float array) 和 user_int_feats (int array)
        处理：对于fid相同的Int和Dense特征进行Concat拼接
        目标：通过Shared MLP将变长或高维向量投影到Transformer隐层维度d
        """
        print("\n=== Dense-to-Token Mapper 处理开始 ===")
        
        # 1. 处理所有数组特征
        processed_features = {}
        
        print(f"\n1. 处理数组特征:")
        print(f"   整型数组特征数: {len(self.user_int_array_features)}")
        print(f"   浮点数组特征数: {len(self.user_dense_array_features)}")
        print(f"   物品数组特征数: {len(self.item_int_array_features)}")
        
        all_array_features = (self.user_int_array_features + 
                             self.user_dense_array_features + 
                             self.item_int_array_features)
        
        for col in all_array_features:
            print(f"   \n处理特征: {col}")
            processed = self._process_array_feature(df, col)
            processed_features[col] = processed
            print(f"     处理后形状: {processed.shape}")
            print(f"     数据类型: {processed.dtype}")
            print(f"     示例值: {processed[0][:3]}..." if processed.shape[1] > 3 else f"     示例值: {processed[0]}")
        
        # 2. 按fid分组拼接特征
        print(f"\n2. 按fid分组拼接特征:")
        fid_tokens = {}  # 归一化后的Token
        fid_tokens_raw = {}  # 原始未归一化的Token
        
        for fid, feats in self.fid_feature_mapping.items():
            print(f"   \n处理FID {fid}:")
            print(f"     包含特征: {feats}")
            
            # 获取该fid下的所有特征
            fid_arrays = []
            for feat in feats:
                if feat in processed_features:
                    fid_arrays.append(processed_features[feat])
            
            if not fid_arrays:
                print(f"     无有效特征")
                continue
            
            # 转换为PyTorch张量并拼接
            # 注意：这里假设所有特征的长度相同，实际情况可能需要更复杂的处理
            fid_tensors = []
            for i, arr in enumerate(fid_arrays):
                tensor = torch.tensor(arr, dtype=torch.float32)
                fid_tensors.append(tensor)
                print(f"     特征 {feats[i]} 张量形状: {tensor.shape}")
            
            # 拼接特征 (batch_size, feature_length * num_features)
            concatenated = torch.cat(fid_tensors, dim=1)
            print(f"     拼接后形状: {concatenated.shape}")
            print(f"     拼接后示例 (前1个样本，前10个值): {concatenated[0, :10].tolist()}")
            
            # 3. 通过Shared MLP投影到Transformer隐层维度
            # 动态调整MLP的输入维度
            current_input_dim = concatenated.shape[1]
            if self.shared_mlp[0].in_features != current_input_dim:
                print(f"     调整MLP输入维度: {self.shared_mlp[0].in_features} -> {current_input_dim}")
                self.shared_mlp[0] = nn.Linear(current_input_dim, self.transformer_hidden_dim)
            
            # 应用MLP
            mapped = self.shared_mlp(concatenated)
            print(f"     MLP投影后形状: {mapped.shape}")
            print(f"     MLP投影后示例 (前1个样本，前5个值): {mapped[0, :5].tolist()}")
            print(f"     均值: {mapped.mean().item():.4f}, 方差: {mapped.var().item():.4f}")
            
            # 应用RMSNorm层
            mapped_norm = self.dense_token_rms_norm(mapped)
            print(f"     RMSNorm后形状: {mapped_norm.shape}")
            print(f"     RMSNorm后示例 (前1个样本，前5个值): {mapped_norm[0, :5].tolist()}")
            print(f"     均值: {mapped_norm.mean().item():.4f}, 方差: {mapped_norm.var().item():.4f}")
            
            fid_tokens[fid] = mapped_norm
        
        # 4. 汇总所有fid的Token
        print(f"\n3. 汇总所有fid的Token:")
        all_tokens = []
        for fid, token in fid_tokens.items():
            all_tokens.append(token)
        
        if all_tokens:
            # 拼接所有Token (batch_size, num_fids * hidden_dim)
            concatenated_tokens = torch.cat(all_tokens, dim=1)
            print(f"   所有Token拼接后形状: {concatenated_tokens.shape}")
            print(f"   拼接后示例 (前1个样本，前10个值): {concatenated_tokens[0, :10].tolist()}")
            print(f"   均值: {concatenated_tokens.mean().item():.4f}, 方差: {concatenated_tokens.var().item():.4f}")
        else:
            concatenated_tokens = None
            print(f"   无有效Token")
        
        print(f"\n=== Dense-to-Token Mapper 处理完成 ===")
        
        # 返回处理结果
        result = {
            'fid_tokens': fid_tokens,  # 归一化后的Token
            'fid_tokens_raw': fid_tokens_raw,  # 原始未归一化的Token
            'all_dense_tokens': concatenated_tokens,
            'processed_array_features': processed_features
        }
        
        return result
    
    def _hash_bucketize_sequence(self, id_value: int, bucket_size: int) -> int:
        """对序列中的ID值进行哈希分桶，避免大ID导致的词汇表过大问题"""
        import hashlib
        
        # 使用SHA256哈希
        hash_val = hashlib.sha256(str(id_value).encode()).hexdigest()
        # 转换为整数并取模得到桶索引
        bucket_idx = int(hash_val, 16) % bucket_size
        
        # 确保桶索引大于SEP Token ID
        return bucket_idx + 1  # SEP Token ID为0，所以桶索引从1开始
    
    def transform_sequence(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """功能三：S-Tokenizer (Sequential Domain Fusion)
        
        输入：Domain A, B, C, D 的行为序列
        处理：
        1. Global Interleaving：假设序列已按时间顺序排列，直接拼接
        2. Truncation & Padding：对超长序列进行截断，对短序列进行补齐
        3. Special Tokens：在不同域的切换处插入可学习的 [SEP] Token
        4. Shared Embedding：通过共享的Embedding层处理ID序列
        
        目标：输出同质化的 S-tokens，共享同一套投影参数
        """
        print("\n=== S-Tokenizer 处理开始 ===")
        
        # 使用哈希分桶来处理大ID
        bucket_size = 1000000  # 设置哈希桶大小
        print(f"   哈希桶大小: {bucket_size}")
        
        # 1. 获取所有序列特征列
        domain_columns = {
            'domain_a': self.domain_sequence_features['domain_a'],
            'domain_b': self.domain_sequence_features['domain_b'],
            'domain_c': self.domain_sequence_features['domain_c'],
            'domain_d': self.domain_sequence_features['domain_d']
        }
        
        print(f"\n1. 序列特征概览:")
        for domain, cols in domain_columns.items():
            print(f"   {domain}: {len(cols)} 个特征")
        
        # 2. 处理每个样本的序列
        batch_size = len(df)
        s_tokens_list = []
        masks_list = []
        
        print(f"\n2. 处理序列数据:")
        print(f"   样本数量: {batch_size}")
        print(f"   最大序列长度: {self.max_seq_length}")
        print(f"   SEP Token ID: {self.sep_token_id}")
        
        for i in range(batch_size):
            if i % 100 == 0:
                print(f"   处理样本 {i}/{batch_size}")
            
            # 收集所有域的序列
            all_sequences = []
            
            for domain, cols in domain_columns.items():
                # 对每个域，将所有序列特征拼接
                domain_seq = []
                for col in cols:
                    seq = df[col].iloc[i]
                    # 处理None值
                    if seq is None:
                        continue
                    # 将numpy数组转换为列表
                    if isinstance(seq, np.ndarray):
                        seq = seq.tolist()
                    # 确保是可迭代的
                    if isinstance(seq, (list, tuple)):
                        # 对序列中的每个ID进行哈希分桶
                        hashed_seq = [self._hash_bucketize_sequence(x, bucket_size) for x in seq if x != 0]
                        domain_seq.extend(hashed_seq)
                
                # 添加域的序列（已经去除填充0）
                if domain_seq:
                    all_sequences.append(domain_seq)
                    # 在域之间添加SEP Token
                    all_sequences.append([self.sep_token_id])
            
            # 移除最后一个多余的SEP Token
            if all_sequences and all_sequences[-1] == [self.sep_token_id]:
                all_sequences.pop()
            
            # 拼接所有序列
            combined_seq = []
            for seq in all_sequences:
                combined_seq.extend(seq)
            
            # 3. 截断或填充序列
            if len(combined_seq) > self.max_seq_length:
                # 截断超长序列
                combined_seq = combined_seq[:self.max_seq_length]
                print(f"     样本 {i}: 序列过长 ({len(combined_seq)})，已截断")
            
            # 创建掩码（1表示有效，0表示填充）
            mask = [1] * len(combined_seq)
            
            # 填充到最大长度
            if len(combined_seq) < self.max_seq_length:
                padding_length = self.max_seq_length - len(combined_seq)
                combined_seq.extend([0] * padding_length)
                mask.extend([0] * padding_length)
            
            s_tokens_list.append(combined_seq)
            masks_list.append(mask)
        
        # 4. 转换为张量
        s_tokens_tensor = torch.tensor(s_tokens_list, dtype=torch.long)
        masks_tensor = torch.tensor(masks_list, dtype=torch.float32)
        
        print(f"\n3. 序列处理完成:")
        print(f"   序列张量形状: {s_tokens_tensor.shape}")
        print(f"   掩码张量形状: {masks_tensor.shape}")
        print(f"   示例序列 (前50个Token): {s_tokens_tensor[0, :50].tolist()}")
        print(f"   示例掩码 (前50个位置): {masks_tensor[0, :50].tolist()}")
        
        # 5. 调整Embedding层的词汇表大小（如果需要）
        if bucket_size + 1 > self.shared_embedding.num_embeddings:
            print(f"\n调整Embedding层词汇表大小: {self.shared_embedding.num_embeddings} -> {bucket_size + 1}")
            new_embedding = nn.Embedding(
                num_embeddings=bucket_size + 1,  # +1是为了包含SEP Token
                embedding_dim=self.embedding_dim,
                padding_idx=0
            )
            # 复制现有权重
            new_embedding.weight.data[:self.shared_embedding.num_embeddings] = self.shared_embedding.weight.data
            self.shared_embedding = new_embedding
        
        # 6. 应用共享Embedding层
        print(f"\n4. 应用共享Embedding层:")
        embedded_tokens = self.shared_embedding(s_tokens_tensor)
        print(f"   Embedding后形状: {embedded_tokens.shape}")
        print(f"   示例嵌入 (前1个Token，前5个维度): {embedded_tokens[0, 0, :5].tolist()}")
        print(f"   均值: {embedded_tokens.mean().item():.4f}, 方差: {embedded_tokens.var().item():.4f}")
        
        print(f"\n=== S-Tokenizer 处理完成 ===")
        
        # 返回处理结果
        result = {
            's_tokens': s_tokens_tensor,
            's_token_masks': masks_tensor,
            'embedded_s_tokens': embedded_tokens
        }
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """拟合并转换数据"""
        self.fit(df)
        return self.transform(df)

# 示例用法
if __name__ == "__main__":
    import os
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OneTransPreprocessor 特征处理工具")
    parser.add_argument('--function', type=str, default='both', choices=['ns_tokenizer', 'dense_token', 'sequence', 'all', 'both'],
                      help='选择要运行的功能: ns_tokenizer(功能一), dense_token(功能二), sequence(功能三), all(三者都运行), both(仅功能一和二)')
    args = parser.parse_args()
    
    print("OneTransPreprocessor 测试")
    print("=" * 50)
    
    # 配置文件路径
    config_path = r"E:\py_project\Tencent_Ad_Algo_OneTrans\config\feature_config.json"
    
    # 数据文件路径
    data_path = r"E:\py_project\Tencent_Ad_Algo_OneTrans\data\raw\demo_1000.parquet"
    
    try:
        # 加载数据
        print(f"\n1. 加载数据文件: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"   数据加载完成，形状: {df.shape}")
        
        # 显示输入数据示例
        print(f"   \n输入数据示例 (前3行，仅显示部分特征):")
        print(df.head(3)[['user_id', 'item_id', 'user_int_feats_1', 'user_int_feats_3', 'user_int_feats_4']])
        
        # 显示特征统计信息
        print(f"   \n输入数据统计信息:")
        print(f"   - 总样本数: {len(df)}")
        print(f"   - 总特征数: {len(df.columns)}")
        print(f"   - 缺失值统计 (前5个特征):")
        for col in df.columns[:5]:
            print(f"     {col}: {df[col].isna().sum()} 个缺失值")
        
        # 创建预处理器实例
        print(f"\n2. 创建OneTransPreprocessor实例")
        preprocessor = OneTransPreprocessor(config_path)
        
        # 初始化结果变量
        ns_result = None
        dense_result = None
        seq_result = None
        
        # 运行选择的功能
        if args.function in ['ns_tokenizer', 'both', 'all']:
            print(f"\n=== 运行功能一: NS-Tokenizer ===")
            # 单独执行fit步骤
            print(f"\n3. 执行fit步骤 (计算特征统计信息)")
            preprocessor.fit(df)
            
            # 单独执行transform步骤
            print(f"\n4. 执行transform步骤 (转换数据)")
            ns_result = preprocessor.transform(df)
            
            # 打印中间结果
            print(f"\n5. 中间过程结果")
            
            # 显示原始特征到桶化后的转换
            print(f"   \n原始特征 -> 桶化特征示例:")
            print(f"   特征 'user_int_feats_1' 原始值: {df['user_int_feats_1'].head(3).tolist()}")
            processed_user_int_feats_1 = df['user_int_feats_1'].apply(
                lambda x: preprocessor._numeric_bucketize(x, preprocessor.num_buckets, 
                                                       preprocessor.numeric_stats['user_int_feats_1']['min'],
                                                       preprocessor.numeric_stats['user_int_feats_1']['max'])
            )
            print(f"   特征 'user_int_feats_1' 桶化后值: {processed_user_int_feats_1.head(3).tolist()}")
            
            # 显示特征矩阵示例
            print(f"   \n特征矩阵 (前2行，前5列):")
            scalar_feats_np = ns_result['scalar_features'].detach().numpy()
            print(scalar_feats_np[:2, :5])
            
            # 显示投影层参数
            print(f"   \n投影层参数示例 (Token 1):")
            print(f"   权重形状: {preprocessor.projection_layers[0].weight.shape}")
            print(f"   权重值 (前5个): {preprocessor.projection_layers[0].weight[0, :5].tolist()}")
            print(f"   偏置值: {preprocessor.projection_layers[0].bias.item()}")
            
            # 显示单个Token的计算过程
            print(f"   \n单个Token计算示例 (Token 1，前2个样本):")
            token1_output = preprocessor.projection_layers[0](ns_result['scalar_features'][:2])
            print(f"   输入特征 (前2个样本，前5列):")
            print(ns_result['scalar_features'][:2, :5])
            print(f"   Token 1输出: {token1_output.tolist()}")
            
            # 打印最终结果
            print(f"\n6. 功能一最终输出结果")
            print(f"   NS-Tokens形状 (归一化后): {ns_result['ns_tokens'].shape}")
            print(f"   NS-Tokens形状 (原始): {ns_result['ns_tokens_raw'].shape}")
            print(f"   标量特征形状: {ns_result['scalar_features'].shape}")
            
            # 显示NS-Tokens示例 (归一化前后对比)
            print(f"   \nNS-Tokens示例 (前2个样本，归一化前后对比):")
            print(f"   原始NS-Tokens:")
            print(ns_result['ns_tokens_raw'][:2])
            print(f"   归一化后NS-Tokens:")
            print(ns_result['ns_tokens'][:2])
            
            # 显示NS-Tokens统计信息 (归一化前后对比)
            print(f"   \nNS-Tokens统计信息对比:")
            print(f"   {'指标':<15} {'原始':<12} {'RMSNorm后':<12}")
            print(f"   {'-'*40}")
            print(f"   {'均值':<15} {ns_result['ns_tokens_raw'].mean().item():<12.4f} {ns_result['ns_tokens'].mean().item():<12.4f}")
            print(f"   {'方差':<15} {ns_result['ns_tokens_raw'].var().item():<12.4f} {ns_result['ns_tokens'].var().item():<12.4f}")
            print(f"   {'最小值':<15} {ns_result['ns_tokens_raw'].min().item():<12.4f} {ns_result['ns_tokens'].min().item():<12.4f}")
            print(f"   {'最大值':<15} {ns_result['ns_tokens_raw'].max().item():<12.4f} {ns_result['ns_tokens'].max().item():<12.4f}")
            
            print(f"\n7. 功能一计算验证")
            # 手动计算第一个样本的第一个Token，验证准确性
            sample_idx = 0
            token_idx = 0
            
            print(f"   \n手动验证样本 {sample_idx} 的 Token {token_idx+1}:")
            
            # 获取输入特征
            input_feat = ns_result['scalar_features'][sample_idx]
            print(f"   输入特征 (前10个): {input_feat[:10].tolist()}")
            
            # 获取投影层参数
            weight = preprocessor.projection_layers[token_idx].weight[0]
            bias = preprocessor.projection_layers[token_idx].bias[0]
            print(f"   投影权重 (前10个): {weight[:10].tolist()}")
            print(f"   偏置值: {bias.item()}")
            
            # 手动计算投影输出
            manual_proj = torch.dot(input_feat, weight) + bias
            print(f"   投影输出: {manual_proj.item():.4f}")
            
            # 手动计算RMSNorm
            print(f"   \n手动验证RMSNorm计算:")
            sample_raw_tokens = ns_result['ns_tokens_raw'][sample_idx]
            print(f"   原始Tokens: {sample_raw_tokens.tolist()}")
            
            # 计算RMS
            rms_manual = torch.sqrt(torch.mean(sample_raw_tokens ** 2) + preprocessor.rms_norm.eps)
            print(f"   计算RMS值: {rms_manual.item():.6f}")
            
            # 归一化
            norm_manual = sample_raw_tokens / rms_manual
            print(f"   归一化后值: {norm_manual.tolist()}")
            
            # 应用权重和偏置
            rms_norm_manual = preprocessor.rms_norm.weight * norm_manual + preprocessor.rms_norm.bias
            print(f"   最终RMSNorm结果: {rms_norm_manual.tolist()}")
            
            # 与模型输出比较
            model_rms_norm = ns_result['ns_tokens'][sample_idx]
            print(f"   模型RMSNorm结果: {model_rms_norm.tolist()}")
            
            # 计算误差
            error = torch.mean(torch.abs(rms_norm_manual - model_rms_norm))
            print(f"   RMSNorm计算误差: {error.item():.10f}")
            print(f"   RMSNorm权重: {preprocessor.rms_norm.weight.tolist()}")
            print(f"   RMSNorm偏置: {preprocessor.rms_norm.bias.tolist()}")
        
        if args.function in ['dense_token', 'both', 'all']:
            print(f"\n=== 运行功能二: Dense-to-Token Mapper ===")
            # 执行Dense-to-Token Mapper
            dense_result = preprocessor.transform_dense_token(df)
            
            # 显示功能二的结果
            print(f"\n8. 功能二最终结果")
            print(f"   FID分组数量: {len(dense_result['fid_tokens'])}")
            print(f"   所有Token拼接后形状: {dense_result['all_dense_tokens'].shape if dense_result['all_dense_tokens'] is not None else 'None'}")
            
            # 显示部分FID的Token结果
            print(f"   \n部分FID的Token结果:")
            for i, (fid, token) in enumerate(list(dense_result['fid_tokens'].items())[:3]):
                print(f"     FID {fid}: 形状 {token.shape}, 均值 {token.mean().item():.4f}, 方差 {token.var().item():.4f}")
                print(f"     示例值 (前1个样本，前5个值): {token[0, :5].tolist()}")
        
        if args.function in ['sequence', 'all']:
            print(f"\n=== 运行功能三: S-Tokenizer ===")
            # 执行S-Tokenizer
            seq_result = preprocessor.transform_sequence(df)
            
            # 显示功能三的结果
            print(f"\n9. 功能三最终结果")
            print(f"   S-Tokens形状: {seq_result['s_tokens'].shape}")
            print(f"   掩码形状: {seq_result['s_token_masks'].shape}")
            print(f"   Embedding后形状: {seq_result['embedded_s_tokens'].shape}")
            
            # 显示序列处理结果
            print(f"   \n序列处理结果示例:")
            print(f"     原始Token序列 (前20个): {seq_result['s_tokens'][0, :20].tolist()}")
            print(f"     掩码 (前20个): {seq_result['s_token_masks'][0, :20].tolist()}")
            print(f"     Embedding结果 (前1个Token，前10个维度): {seq_result['embedded_s_tokens'][0, 0, :10].tolist()}")
            
            # 统计有效序列长度
            valid_lengths = seq_result['s_token_masks'].sum(dim=1)
            print(f"   \n序列长度统计:")
            print(f"     平均有效长度: {valid_lengths.mean().item():.2f}")
            print(f"     最大有效长度: {valid_lengths.max().item()}")
            print(f"     最小有效长度: {valid_lengths.min().item()}")
        
        # 保存处理结果
        print(f"\n=== 保存处理结果 ===")
        import os
        
        # 创建保存目录
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        print(f"   保存目录: {processed_dir}")
        
        # 保存功能一结果
        if args.function in ['ns_tokenizer', 'both', 'all']:
            ns_save_path = os.path.join(processed_dir, 'ns_tokens.pt')
            torch.save(ns_result, ns_save_path)
            print(f"   功能一结果保存至: {ns_save_path}")
        
        # 保存功能二结果
        if args.function in ['dense_token', 'both', 'all']:
            dense_save_path = os.path.join(processed_dir, 'dense_tokens.pt')
            torch.save(dense_result, dense_save_path)
            print(f"   功能二结果保存至: {dense_save_path}")
        
        # 保存功能三结果
        if args.function in ['sequence', 'all']:
            seq_save_path = os.path.join(processed_dir, 'sequence_tokens.pt')
            torch.save(seq_result, seq_save_path)
            print(f"   功能三结果保存至: {seq_save_path}")
        
        print(f"\n=== 所有功能运行完成 ===")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()