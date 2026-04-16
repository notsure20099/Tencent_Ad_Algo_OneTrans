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
        labels = (subset_df[cfg['label_col']].to_numpy() == 1).astype(float)
        
        for i in tqdm(range(len(subset_df)), desc=desc):
            # 创新点 1：序列（S）在前
            s_tokens = []
            for domain_name, cols in cfg['s_features'].items():
                for col in cols:
                    raw_seq = subset_df[col][i]
                    mapped = [id_map.get(str(int(x)), PAD_ID) for x in (raw_seq or [])]
                    mapped = mapped[-cfg['s_max_len']:]
                    mapped = mapped + [PAD_ID] * (cfg['s_max_len'] - len(mapped))
                    s_tokens.extend(mapped)
                s_tokens.append(SEP_ID) 

            # 非序列（NS）在后
            ns_tokens = []
            ns_cols = cfg['ns_features']['user_scalar'] + cfg['ns_features']['item_scalar']
            for col in ns_cols:
                val = subset_df[col][i]
                token = id_map.get(str(int(val)), PAD_ID) if val is not None else PAD_ID
                ns_tokens.append(token)
            
            # [S_Tokens, NS_Tokens]
            all_features.append(s_tokens + ns_tokens)

        return torch.tensor(all_features, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

    os.makedirs(output_dir, exist_ok=True)
    split_idx = int(len(df) * 0.8)
    
    x_train, y_train = process_to_tensor(df.head(split_idx), "Training Data")
    torch.save({'x': x_train, 'y': y_train}, os.path.join(output_dir, 'train_data.pt'))
    
    x_val, y_val = process_to_tensor(df.tail(len(df) - split_idx), "Val Data")
    torch.save({'x': x_val, 'y': y_val}, os.path.join(output_dir, 'val_data.pt'))