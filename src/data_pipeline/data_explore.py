import os
import sys

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.utils import (
    FeatureConfigManager,
    DataIO,
    DataStatistics,
    DataPreprocessorUtils,
    FeatureExplorer
)

class DataPreprocessor:
    def __init__(self, config_path):
        """初始化预处理类，加载特征配置"""
        self.config_manager = FeatureConfigManager(config_path)
        self.processed_data = None
    
    def load_data(self, data_path):
        """加载原始数据"""
        self.processed_data = DataIO.load_data(data_path)
        return self
    
    def get_feature_columns(self, feature_type):
        """根据特征类型获取对应的列名列表"""
        return self.config_manager.get_feature_columns(feature_type)
    
    def basic_statistics(self):
        """计算数据的基本统计信息"""
        if self.processed_data is None:
            raise ValueError("数据尚未加载，请先调用load_data方法")
        
        return DataStatistics.calculate_basic_statistics(
            self.processed_data, 
            self.config_manager
        )
    
    def preprocess(self, fillna=True):
        """执行数据预处理"""
        if self.processed_data is None:
            raise ValueError("数据尚未加载，请先调用load_data方法")
        
        print("开始数据预处理...")
        
        # 1. 处理缺失值
        if fillna:
            self.processed_data = DataPreprocessorUtils.process_missing_values(
                self.processed_data, 
                self.config_manager
            )
        
        # 2. 确保数据类型正确
        self.processed_data = DataPreprocessorUtils.check_data_types(
            self.processed_data, 
            self.config_manager
        )
        
        print("预处理完成！")
        return self
    
    def save_processed_data(self, output_path):
        """保存处理后的数据"""
        if self.processed_data is None:
            raise ValueError("没有可保存的数据，请先加载并预处理数据")
        
        return DataIO.save_data(self.processed_data, output_path)
    
    def get_data(self):
        """获取处理后的数据"""
        if self.processed_data is None:
            raise ValueError("数据尚未加载或处理")
        return self.processed_data
    
    def explore_features(self):
        """探索所有特征"""
        if self.processed_data is None:
            raise ValueError("数据尚未加载，请先调用load_data方法")
        
        # 使用FeatureExplorer进行特征探索
        return FeatureExplorer.explore_all_features(
            self.processed_data, 
            self.config_manager
        )
    
    def print_feature_exploration(self, feature_explorations):
        """打印特征探索结果"""
        FeatureExplorer.print_feature_exploration_results(feature_explorations)

import argparse

def main(explore=False, preprocess=False, save_results=False):
    # 配置文件路径
    config_path = r"E:\py_project\Tencent_Ad_Algo_OneTrans\config\feature_config.json"
    
    # 数据文件路径
    data_path = r"E:\py_project\Tencent_Ad_Algo_OneTrans\data\raw\demo_1000.parquet"
    
    # 输出文件路径
    output_path = r"E:\py_project\Tencent_Ad_Algo_OneTrans\data\processed\processed_data.parquet"
    
    # 创建预处理实例
    preprocessor = DataPreprocessor(config_path)
    
    # 加载数据
    preprocessor.load_data(data_path)
    
    # 查看基本统计信息
    print("=== 数据基本统计信息 ===")
    stats = preprocessor.basic_statistics()
    print(f"总样本数: {stats['total_rows']}")
    print(f"总列数: {stats['total_columns']}")
    print(f"ID和标签列数: {len(stats['id_label_columns']['columns'])}")
    print(f"用户整数特征数: {stats['user_int_features']['scalar_columns']['count']} (标量) + {stats['user_int_features']['array_columns']['count']} (数组)")
    print(f"用户密集特征数: {stats['user_dense_features']['count']}")
    print(f"物品整数特征数: {stats['item_int_features']['scalar_columns']['count']} (标量) + {stats['item_int_features']['array_columns']['count']} (数组)")
    print(f"域序列特征数: {stats['domain_sequence_features']['count']}")
    
    # 特征探索（可选项）
    feature_explorations = None
    if explore:
        feature_explorations = preprocessor.explore_features()
        preprocessor.print_feature_exploration(feature_explorations)
        
        if save_results:
            from src.utils import FeatureExplorer
            FeatureExplorer.save_exploration_to_config(feature_explorations, config_path)
    
    # 执行预处理（可选项）
    if preprocess:
        preprocessor.preprocess(fillna=True)
        
        # 保存处理后的数据
        preprocessor.save_processed_data(output_path)
        
        print("\n=== 预处理完成 ===")
        print(f"原始数据形状: {preprocessor.get_data().shape}")
        print(f"处理后的数据已保存到: {output_path}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="数据探索和预处理工具")
    parser.add_argument('--explore', action='store_true', help='运行特征探索')
    parser.add_argument('--preprocess', action='store_true', help='运行数据预处理')
    parser.add_argument('--save-results', action='store_true', help='保存特征探索结果到配置文件')
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，显示帮助信息
    if not args.explore and not args.preprocess:
        parser.print_help()
        print("\n示例用法:")
        print("  python src/data_pipeline/data_explore.py --explore  # 仅运行特征探索")
        print("  python src/data_pipeline/data_explore.py --preprocess  # 仅运行数据预处理")
        print("  python src/data_pipeline/data_explore.py --explore --preprocess --save-results  # 运行所有功能")
    else:
        main(explore=args.explore, preprocess=args.preprocess, save_results=args.save_results)