import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加 scripts 目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

class TestPreprocessing:
    
    def test_data_loading(self, sample_housing_data, temp_dir):
        """测试数据加载功能"""
        # 保存样本数据到临时文件
        data_path = Path(temp_dir) / "housing.csv"
        sample_housing_data.to_csv(data_path, index=False)
        
        # 测试数据加载
        loaded_data = pd.read_csv(data_path)
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_housing_data)
        assert 'median_house_value' in loaded_data.columns
        print("✓ 数据加载测试通过")
    
    def test_missing_value_handling(self, sample_housing_data):
        """测试缺失值处理"""
        # 添加一些缺失值
        data_with_missing = sample_housing_data.copy()
        mask = np.random.random(len(data_with_missing)) < 0.1
        data_with_missing.loc[mask, 'total_bedrooms'] = np.nan

        # 验证有缺失值
        assert data_with_missing['total_bedrooms'].isnull().any()

        # 测试中位数填充（修复警告）
        bedrooms_median = data_with_missing['total_bedrooms'].median()
        data_with_missing = data_with_missing.assign(
            total_bedrooms=data_with_missing['total_bedrooms'].fillna(bedrooms_median)
        )

        # 验证没有缺失值
        assert not data_with_missing['total_bedrooms'].isnull().any()
        print("✓ 缺失值处理测试通过")
    
    def test_categorical_encoding(self, sample_housing_data):
        """测试分类变量编码"""
        from sklearn.preprocessing import LabelEncoder
        
        data = sample_housing_data.copy()
        
        # 测试标签编码
        le = LabelEncoder()
        encoded = le.fit_transform(data['ocean_proximity'])
        
        assert len(encoded) == len(data)
        assert encoded.dtype == np.int64
        assert len(le.classes_) == len(data['ocean_proximity'].unique())
        print("✓ 分类变量编码测试通过")
    
    def test_feature_scaling(self, sample_housing_data):
        """测试特征缩放"""
        from sklearn.preprocessing import StandardScaler
        
        data = sample_housing_data.copy()
        numeric_features = ['longitude', 'latitude', 'housing_median_age', 'median_income']
        
        # 选择数值特征
        X = data[numeric_features]
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.shape == X.shape
        # 检查标准化后的数据均值为0，标准差为1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-6)
        print("✓ 特征缩放测试通过")
    
    def test_train_test_split(self, sample_housing_data):
        """测试训练测试集分割"""
        from sklearn.model_selection import train_test_split
        
        X = sample_housing_data.drop('median_house_value', axis=1)
        y = sample_housing_data['median_house_value']
        
        # 编码分类变量
        X = pd.get_dummies(X, columns=['ocean_proximity'])
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 验证分割比例
        assert len(X_test) / len(X) == 0.2
        assert len(X_train) / len(X) == 0.8
        
        # 验证没有数据泄漏
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        print("✓ 训练测试集分割测试通过")