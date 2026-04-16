import polars as pl
import json
import os
import torch
import yaml
from tqdm import tqdm

def prepare_data(input_path, mapping_path, output_dir, config_path):
    with open(config_path, 'r') as f:
        full_cfg = yaml.safe_load(f)
        cfg = full_cfg['feature_schema']
    
    df = pl.read_parquet(input_path)
    with open(mapping_path, 'r') as f:
        id_map = json.load(f)

    PAD_ID, SEP_ID = 0, 1

    def process_to_tensor(subset_df, desc):
        all_features = []
        labels = (subset_df[cfg['label_col']].to_numpy() == full_cfg['train']['positive_label']).astype(float)
        
        for i in tqdm(range(len(subset_df)), desc=desc):
            # --- 创新点 1: 序列在前 (S-Tokens)，非序列在后 (NS-Tokens) ---
            s_tokens = []
            for domain_name, cols in cfg['s_features'].items():
                for col in cols:
                    raw_seq = subset_df[col][i]
                    mapped = [id_map.get(str(int(x)), PAD_ID) for x in (raw_seq or [])]
                    mapped = mapped[-cfg['s_max_len']:] # 截断
                    mapped = mapped + [PAD_ID] * (cfg['s_max_len'] - len(mapped)) # 填充
                    s_tokens.extend(mapped)
                s_tokens.append(SEP_ID) 

            ns_tokens = []
            for col in (cfg['ns_features']['user_scalar'] + cfg['ns_features']['item_scalar']):
                val = subset_df[col][i]
                ns_tokens.append(id_map.get(str(int(val)), PAD_ID) if val is not None else PAD_ID)
            
            # 拼接顺序：[S, NS] -> 方便后续 KV Cache 只缓存 S 部分
            unified_tokens = s_tokens + ns_tokens
            all_features.append(unified_tokens)

        return torch.tensor(all_features, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

    os.makedirs(output_dir, exist_ok=True)
    x_train, y_train = process_to_tensor(df.head(int(len(df)*0.8)), "训练集转换")
    torch.save({'x': x_train, 'y': y_train}, os.path.join(output_dir, 'train_data.pt'))
    
    x_val, y_val = process_to_tensor(df.tail(int(len(df)*0.2)), "验证集转换")
    torch.save({'x': x_val, 'y': y_val}, os.path.join(output_dir, 'val_data.pt'))