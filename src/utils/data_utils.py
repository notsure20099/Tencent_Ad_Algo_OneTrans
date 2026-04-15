import pandas as pd
import numpy as np
import json
import os
from collections import Counter

class FeatureConfigManager:
    """特征配置管理器"""
    def __init__(self, config_path):
        self.config_path = config_path
        self.feature_config = self.load_config()
    
    def load_config(self):
        """加载特征配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_feature_columns(self, feature_type):
        """根据特征类型获取对应的列名列表"""
        config = self.feature_config
        
        if feature_type == 'id_and_label':
            return config['id_and_label_columns']['columns']
        elif feature_type == 'user_int_scalar':
            return config['user_int_features']['scalar_columns']
        elif feature_type == 'user_int_array':
            return config['user_int_features']['array_columns']
        elif feature_type == 'user_dense':
            return config['user_dense_features']['array_columns']
        elif feature_type == 'item_int_scalar':
            return config['item_int_features']['scalar_columns']
        elif feature_type == 'item_int_array':
            return config['item_int_features']['array_columns']
        elif feature_type == 'domain_sequence':
            # 合并所有域的序列特征
            all_seq_cols = []
            for domain in ['domain_a', 'domain_b', 'domain_c', 'domain_d']:
                all_seq_cols.extend(config['domain_sequence_features'][domain])
            return all_seq_cols
        elif feature_type == 'all':
            # 返回所有特征列
            all_cols = config['id_and_label_columns']['columns']
            all_cols.extend(config['user_int_features']['scalar_columns'])
            all_cols.extend(config['user_int_features']['array_columns'])
            all_cols.extend(config['user_dense_features']['array_columns'])
            all_cols.extend(config['item_int_features']['scalar_columns'])
            all_cols.extend(config['item_int_features']['array_columns'])
            for domain in ['domain_a', 'domain_b', 'domain_c', 'domain_d']:
                all_cols.extend(config['domain_sequence_features'][domain])
            return all_cols
        else:
            raise ValueError(f"未知的特征类型: {feature_type}")

class DataIO:
    """数据输入输出工具"""
    
    @staticmethod
    def load_data(data_path):
        """加载原始数据"""
        print(f"加载数据文件: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"数据加载完成，形状: {df.shape}")
        print(f"数据列数量: {len(df.columns)}")
        return df
    
    @staticmethod
    def save_data(df, output_path):
        """保存数据"""
        print(f"保存数据到: {output_path}")
        df.to_parquet(output_path, index=False)
        print(f"数据保存完成，文件大小: {os.path.getsize(output_path)} 字节")
        return df

class DataStatistics:
    """数据统计工具"""
    
    @staticmethod
    def calculate_basic_statistics(df, config_manager):
        """计算数据的基本统计信息"""
        stats = {}
        
        # 基本信息
        stats['total_rows'] = len(df)
        stats['total_columns'] = len(df.columns)
        
        # ID和标签列统计
        id_label_cols = config_manager.get_feature_columns('id_and_label')
        stats['id_label_columns'] = {
            'columns': id_label_cols,
            'null_counts': df[id_label_cols].isnull().sum().to_dict()
        }
        
        # 用户整数特征统计
        user_int_scalar = config_manager.get_feature_columns('user_int_scalar')
        user_int_array = config_manager.get_feature_columns('user_int_array')
        stats['user_int_features'] = {
            'scalar_columns': {
                'count': len(user_int_scalar),
                'null_counts': df[user_int_scalar].isnull().sum().to_dict()
            },
            'array_columns': {
                'count': len(user_int_array),
                'null_counts': df[user_int_array].isnull().sum().to_dict()
            }
        }
        
        # 用户密集特征统计
        user_dense = config_manager.get_feature_columns('user_dense')
        stats['user_dense_features'] = {
            'count': len(user_dense),
            'null_counts': df[user_dense].isnull().sum().to_dict()
        }
        
        # 物品整数特征统计
        item_int_scalar = config_manager.get_feature_columns('item_int_scalar')
        item_int_array = config_manager.get_feature_columns('item_int_array')
        stats['item_int_features'] = {
            'scalar_columns': {
                'count': len(item_int_scalar),
                'null_counts': df[item_int_scalar].isnull().sum().to_dict()
            },
            'array_columns': {
                'count': len(item_int_array),
                'null_counts': df[item_int_array].isnull().sum().to_dict()
            }
        }
        
        # 域序列特征统计
        domain_seq = config_manager.get_feature_columns('domain_sequence')
        stats['domain_sequence_features'] = {
            'count': len(domain_seq),
            'null_counts': df[domain_seq].isnull().sum().to_dict()
        }
        
        return stats

class FeatureExplorer:
    """特征探索工具"""
    
    @staticmethod
    def explore_single_value_feature(df, col):
        """探索单一值特征"""
        feature_info = {
            'column_name': col,
            'feature_type': 'single_value',
            'data_type': str(df[col].dtype),
            'missing_count': int(df[col].isnull().sum()),
            'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
        }
        
        # 检查是否包含numpy数组
        has_ndarray = False
        try:
            # 尝试获取第一个非空值并检查类型
            first_non_null = df[col].dropna().iloc[0]
            if isinstance(first_non_null, np.ndarray):
                has_ndarray = True
        except (IndexError, ValueError):
            pass
        
        # 计算基本统计信息
        if pd.api.types.is_numeric_dtype(df[col]) and not has_ndarray:
            # 计算数值统计
            feature_info['distribution'] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'unique_values': int(df[col].nunique())
            }
            
            # 尝试计算数值型特征的top 10值分布
            try:
                value_counts = df[col].value_counts().head(10).to_dict()
                feature_info['distribution']['top_10_values'] = value_counts
            except (TypeError, ValueError):
                pass
        else:
            # 非数值型特征或包含numpy数组的特征
            try:
                # 尝试计算唯一值数量
                unique_values = int(df[col].nunique())
                
                # 尝试计算值分布（如果可能）
                try:
                    value_counts = df[col].value_counts().head(10).to_dict()
                    feature_info['distribution'] = {
                        'unique_values': unique_values,
                        'top_10_values': value_counts
                    }
                except (TypeError, ValueError):
                    # 如果无法计算值分布，只记录唯一值数量
                    feature_info['distribution'] = {
                        'unique_values': unique_values,
                        'note': '无法计算值分布（可能包含不可哈希类型）'
                    }
            except (TypeError, ValueError):
                # 如果无法计算唯一值数量，记录错误信息
                feature_info['distribution'] = {
                    'note': '无法计算统计信息（可能包含不可哈希类型）'
                }
        
        return feature_info
    
    @staticmethod
    def explore_array_feature(df, col):
        """探索数组值特征"""
        # 计算缺失值（空列表或NaN）
        def is_empty(x):
            # 检查是否为标量NaN
            if pd.api.types.is_scalar(x):
                return pd.isna(x)
            # 检查是否为列表或数组
            elif isinstance(x, (list, np.ndarray)):
                return len(x) == 0
            return False
        
        missing_count = int(df[col].apply(is_empty).sum())
        
        # 计算数组长度分布
        def get_length(x):
            # 检查是否为标量NaN
            if pd.api.types.is_scalar(x):
                return 0 if pd.isna(x) else 1
            # 检查是否为列表或数组
            elif isinstance(x, (list, np.ndarray)):
                return len(x)
            return 0
        
        lengths = df[col].apply(get_length)
        length_counts = Counter(lengths)
        
        feature_info = {
            'column_name': col,
            'feature_type': 'array_value',
            'missing_count': missing_count,
            'missing_percentage': float(missing_count / len(df) * 100),
            'length_distribution': dict(length_counts),
            'length_statistics': {
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'mean_length': float(lengths.mean()),
                'median_length': float(lengths.median())
            }
        }
        
        return feature_info
    
    @staticmethod
    def explore_all_features(df, config_manager):
        """探索所有特征"""
        print("开始特征探索...")
        
        # 分类特征列（去重）
        single_value_cols = list(set(
            config_manager.get_feature_columns('id_and_label') +
            config_manager.get_feature_columns('user_int_scalar') +
            config_manager.get_feature_columns('item_int_scalar')
        ))
        
        array_value_cols = list(set(
            config_manager.get_feature_columns('user_int_array') +
            config_manager.get_feature_columns('user_dense') +
            config_manager.get_feature_columns('item_int_array') +
            config_manager.get_feature_columns('domain_sequence')
        ))
        
        # 只保留数据中存在的列
        single_value_cols = [col for col in single_value_cols if col in df.columns]
        array_value_cols = [col for col in array_value_cols if col in df.columns]
        
        # 探索所有特征
        feature_explorations = {
            'single_value_features': [],
            'array_value_features': []
        }
        
        # 探索单一值特征
        print(f"探索 {len(single_value_cols)} 个单一值特征...")
        for col in single_value_cols:
            if col in df.columns:
                feature_info = FeatureExplorer.explore_single_value_feature(df, col)
                feature_explorations['single_value_features'].append(feature_info)
        
        # 探索数组值特征
        print(f"探索 {len(array_value_cols)} 个数组值特征...")
        for col in array_value_cols:
            if col in df.columns:
                feature_info = FeatureExplorer.explore_array_feature(df, col)
                feature_explorations['array_value_features'].append(feature_info)
        
        print("特征探索完成！")
        return feature_explorations
    
    @staticmethod
    def print_feature_exploration_results(feature_explorations):
        """打印特征探索结果"""
        print("\n=== 特征探索结果 ===")
        
        # 单一值特征统计
        single_features = feature_explorations['single_value_features']
        print(f"\n单一值特征 ({len(single_features)}):")
        print("-" * 50)
        print(f"{'特征名称':<25} {'数据类型':<10} {'缺失值':<10} {'缺失率(%)':<10}")
        print("-" * 50)
        
        # 用于存储需要展示详细分布的特征
        single_features_with_distribution = []
        
        for feature in single_features:
            print(f"{feature['column_name']:<25} {feature['data_type']:<10} {feature['missing_count']:<10} {feature['missing_percentage']:<10.2f}")
            # 检查是否有分布信息
            if 'distribution' in feature and 'top_10_values' in feature['distribution']:
                single_features_with_distribution.append(feature)
        
        # 数组值特征统计
        array_features = feature_explorations['array_value_features']
        print(f"\n数组值特征 ({len(array_features)}):")
        print("-" * 60)
        print(f"{'特征名称':<25} {'缺失值':<10} {'缺失率(%)':<10} {'平均长度':<10}")
        print("-" * 60)
        
        # 用于存储需要展示长度分布的特征
        array_features_with_length = []
        
        for feature in array_features:
            print(f"{feature['column_name']:<25} {feature['missing_count']:<10} {feature['missing_percentage']:<10.2f} {feature['length_statistics']['mean_length']:<10.2f}")
            array_features_with_length.append(feature)
        
        # 详细信息
        print(f"\n详细信息:")
        print("-" * 30)
        print(f"单一值特征总数: {len(single_features)}")
        print(f"数组值特征总数: {len(array_features)}")
        print(f"总特征数: {len(single_features) + len(array_features)}")
        
        # 缺失值统计
        single_missing_total = sum(f['missing_count'] for f in single_features)
        array_missing_total = sum(f['missing_count'] for f in array_features)
        print(f"\n缺失值统计:")
        print(f"单一值特征总缺失值: {single_missing_total}")
        print(f"数组值特征总缺失值: {array_missing_total}")
        print(f"总缺失值: {single_missing_total + array_missing_total}")
        
        # 打印单一值特征的数据分布（前5个特征）
        print(f"\n=== 单一值特征数据分布 (显示前5个特征) ===")
        for i, feature in enumerate(single_features_with_distribution[:5]):
            print(f"\n{feature['column_name']} (数据类型: {feature['data_type']}):")
            print("-" * 40)
            
            if 'top_10_values' in feature['distribution']:
                top_values = feature['distribution']['top_10_values']
                print(f"值分布 (前{len(top_values)}个值):")
                for value, count in top_values.items():
                    print(f"  {value}: {count} 个")
            
            # 打印统计信息
            if 'min' in feature['distribution']:
                dist = feature['distribution']
                print(f"数值统计: 最小值={dist['min']:.2f}, 最大值={dist['max']:.2f}, 平均值={dist['mean']:.2f}")
        
        # 打印数组值特征的长度分布（前5个特征）
        print(f"\n=== 数组值特征长度分布 (显示前5个特征) ===")
        for i, feature in enumerate(array_features_with_length[:5]):
            print(f"\n{feature['column_name']}:")
            print("-" * 40)
            
            # 打印长度统计
            length_stats = feature['length_statistics']
            print(f"长度统计: 最小长度={length_stats['min_length']}, 最大长度={length_stats['max_length']}, 平均长度={length_stats['mean_length']:.2f}")
            
            # 打印长度分布（前10个长度）
            length_dist = feature['length_distribution']
            # 将长度分布按长度排序
            sorted_lengths = sorted(length_dist.items(), key=lambda x: int(x[0]))
            print(f"长度分布 (前{min(10, len(sorted_lengths))}个长度):")
            for length, count in sorted_lengths[:10]:
                print(f"  长度 {length}: {count} 个")
        
        if len(single_features_with_distribution) > 5:
            print(f"\n... 还有 {len(single_features_with_distribution) - 5} 个单一值特征的分布信息未显示")
        
        if len(array_features_with_length) > 5:
            print(f"... 还有 {len(array_features_with_length) - 5} 个数组值特征的长度分布信息未显示")
    
    @staticmethod
    def save_exploration_to_config(feature_explorations, config_path):
        """将特征探索结果保存到配置文件"""
        print(f"\n将特征探索结果保存到配置文件: {config_path}")
        
        # 读取现有配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 为配置添加探索结果字段
        config['feature_exploration'] = feature_explorations
        
        # 保存更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("特征探索结果已成功保存到配置文件！")
        return config

class DataPreprocessorUtils:
    """数据预处理工具"""
    
    @staticmethod
    def safe_fill_na(x):
        """安全地填充缺失值"""
        # 检查是否为标量NaN
        if pd.api.types.is_scalar(x):
            if pd.isna(x):
                return []
            else:
                return [x]
        # 检查是否为列表
        elif isinstance(x, list):
            return x if x else []
        # 检查是否为numpy数组
        elif isinstance(x, np.ndarray):
            # 检查数组是否为空或只包含NaN
            if x.size == 0 or np.isnan(x).all():
                return []
            else:
                return x.tolist()
        # 其他情况
        else:
            try:
                return list(x)
            except:
                return []
    
    @staticmethod
    def process_missing_values(df, config_manager):
        """处理缺失值"""
        print("处理缺失值...")
        # 数值型特征填充0
        numeric_cols = (config_manager.get_feature_columns('user_int_scalar') + 
                       config_manager.get_feature_columns('item_int_scalar') +
                       config_manager.get_feature_columns('id_and_label'))
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # 数组型特征填充空列表
        array_cols = (config_manager.get_feature_columns('user_int_array') + 
                     config_manager.get_feature_columns('user_dense') +
                     config_manager.get_feature_columns('item_int_array') +
                     config_manager.get_feature_columns('domain_sequence'))
        for col in array_cols:
            df[col] = df[col].apply(DataPreprocessorUtils.safe_fill_na)
        
        return df
    
    @staticmethod
    def check_data_types(df, config_manager):
        """确保数据类型正确"""
        print("检查数据类型...")
        id_label_config = config_manager.feature_config['id_and_label_columns']
        for col, dtype in zip(id_label_config['columns'], id_label_config['data_types']):
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        return df