import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

@pytest.fixture
def sample_housing_data():
    """生成样本房价数据"""
    np.random.seed(42)
    
    n_samples = 100
    data = {
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32, 42, n_samples),
        'housing_median_age': np.random.randint(1, 50, n_samples),
        'total_rooms': np.random.uniform(100, 3000, n_samples),
        'total_bedrooms': np.random.uniform(50, 1500, n_samples),
        'population': np.random.randint(100, 3000, n_samples),
        'households': np.random.uniform(50, 1500, n_samples),
        'median_income': np.random.uniform(1, 15, n_samples),
        'median_house_value': np.random.uniform(100000, 500000, n_samples),
        'ocean_proximity': np.random.choice(['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN'], n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def processed_data(sample_housing_data):
    """返回处理后的数据"""
    data = sample_housing_data.copy()
    
    # 处理缺失值（修复警告）
    data = data.assign(
        total_bedrooms=data['total_bedrooms'].fillna(data['total_bedrooms'].median())
    )
    
    # 编码分类变量
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['ocean_proximity_encoded'] = le.fit_transform(data['ocean_proximity'])
    
    # 选择特征
    feature_columns = [
        'longitude', 'latitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income', 'ocean_proximity_encoded'
    ]
    
    X = data[feature_columns]
    y = data['median_house_value']
    
    return X, y