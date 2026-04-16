import polars as pl
import json
import os
import torch
import yaml
from tqdm import tqdm

def prepare_data(input_path, mapping_path, output_dir, config_path):
    # 1. 加载 YAML 配置
    with open(config_path, 'r') as f:
        full_cfg = yaml.safe_load(f)
        cfg = full_cfg['feature_schema']
    
    # 2. 读取数据与映射表
    print(f"正在读取数据: {input_path}")
    df = pl.read_parquet(input_path)
    with open(mapping_path, 'r') as f:
        id_map = json.load(f)

    PAD_ID, SEP_ID = 0, 1

    def process_to_tensor(subset_df, desc):
        all_features = []
        labels = (subset_df[cfg['label_col']].to_numpy() == full_cfg['train']['positive_label']).astype(float)
        
        for i in tqdm(range(len(subset_df)), desc=desc):
            unified_tokens = []
            
            # --- 链路透视：展示单条数据转换过程 ---
            is_debug = (i == 0)
            if is_debug:
                print(f"\n{'='*20} OneTrans 数据链路透视 (样本 0) {'='*20}")

            # --- 节点 A: 处理非序列特征 (NS-Tokens) ---
            ns_count = 0
            # A1: 处理标量特征
            for col in (cfg['ns_features']['user_scalar'] + cfg['ns_features']['item_scalar']):
                val = subset_df[col][i]
                token = id_map.get(str(int(val)), PAD_ID) if val is not None else PAD_ID
                unified_tokens.append(token)
                ns_count += 1
            
            # A2: 处理数组特征 (取最后一位)
            for col in (cfg['ns_features']['user_array'] + cfg['ns_features']['item_array']):
                arr = subset_df[col][i]
                token = id_map.get(str(int(arr[-1])), PAD_ID) if (arr is not None and len(arr) > 0) else PAD_ID
                unified_tokens.append(token)
                ns_count += 1
            
            unified_tokens.append(SEP_ID) # NS 结束标识
            if is_debug:
                print(f"节点 A (NS-Tokens): 成功将 {ns_count} 个单值特征转换为 Token，末尾已加 SEP(1)")

            # --- 节点 B: 处理序列特征 (S-Tokens) ---
            s_count = 0
            for domain_name, cols in cfg['s_features'].items():
                for col in cols:
                    raw_seq = subset_df[col][i]
                    if raw_seq is None:
                        mapped = [PAD_ID] * cfg['s_max_len']
                    else:
                        mapped = [id_map.get(str(int(x)), PAD_ID) for x in raw_seq if x is not None]
                        # 截断填充
                        mapped = mapped[-cfg['s_max_len']:]
                        mapped = mapped + [PAD_ID] * (cfg['s_max_len'] - len(mapped))
                    
                    unified_tokens.extend(mapped)
                    s_count += len(mapped)
                
                unified_tokens.append(SEP_ID) # Domain 结束标识
            
            if is_debug:
                print(f"节点 B (S-Tokens): 成功将 45 个序列域转换为 Token 流，当前序列总长: {len(unified_tokens)}")
                print(f"节点 C (Final): 样表现形态: {unified_tokens[:10]} ... {unified_tokens[-5:]}")
                print(f"{'='*60}\n")
            
            all_features.append(unified_tokens)

        return torch.tensor(all_features, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

    # 保存逻辑
    os.makedirs(output_dir, exist_ok=True)
    x_train, y_train = process_to_tensor(df.head(int(len(df)*0.8)), "训练集转换")
    torch.save({'x': x_train, 'y': y_train}, os.path.join(output_dir, 'train_data.pt'))
    
    x_val, y_val = process_to_tensor(df.tail(int(len(df)*0.2)), "验证集转换")
    torch.save({'x': x_val, 'y': y_val}, os.path.join(output_dir, 'val_data.pt'))

if __name__ == "__main__":
    # 请根据实际路径调整
    prepare_data(
        input_path="/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/raw/demo_1000.parquet",
        mapping_path="/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/id_mapping.json",
        output_dir="/mnt/workspace/Tencent_Ad_Algo_OneTrans/data/processed",
        config_path="/mnt/workspace/Tencent_Ad_Algo_OneTrans/config/config.yaml"
    )