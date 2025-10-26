import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def main():
    # 创建数据目录
    os.makedirs('data/processed', exist_ok=True)
    
    # 读取数据
    data = pd.read_csv('data/raw/housing.csv')
    print(f"数据加载成功，形状: {data.shape}")
    print("数据列名:", data.columns.tolist())
    print("\n数据前5行:")
    print(data.head())
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(data.isnull().sum())
    
    # 处理缺失值 - 用中位数填充 total_bedrooms
    if 'total_bedrooms' in data.columns:
        bedrooms_median = data['total_bedrooms'].median()
        data['total_bedrooms'].fillna(bedrooms_median, inplace=True)
        print(f"已用中位数 {bedrooms_median} 填充 total_bedrooms 的缺失值")
    
    # 对分类变量 ocean_proximity 进行编码
    label_encoders = {}
    if 'ocean_proximity' in data.columns:
        print(f"\nocean_proximity 的唯一值: {data['ocean_proximity'].unique()}")
        
        # 创建标签编码器
        le = LabelEncoder()
        data['ocean_proximity_encoded'] = le.fit_transform(data['ocean_proximity'])
        label_encoders['ocean_proximity'] = le
        
        print("编码映射:")
        for i, category in enumerate(le.classes_):
            print(f"  {category} -> {i}")
    
    # 准备特征和目标变量
    feature_columns = [
        'longitude', 'latitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income'
    ]
    
    # 添加编码后的分类特征
    if 'ocean_proximity_encoded' in data.columns:
        feature_columns.append('ocean_proximity_encoded')
    
    # 确保所有特征列都存在
    feature_columns = [col for col in feature_columns if col in data.columns]
    
    X = data[feature_columns]
    y = data['median_house_value']
    
    print(f"\n使用的特征: {feature_columns}")
    print(f"目标变量: median_house_value")
    print(f"特征矩阵形状: {X.shape}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化数值特征 (排除分类变量)
    numeric_features = [
        'longitude', 'latitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income'
    ]
    numeric_features = [col for col in numeric_features if col in X.columns]
    
    scaler = StandardScaler()
    
    # 标准化训练集
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    
    # 标准化测试集
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    print(f"标准化了 {len(numeric_features)} 个数值特征")
    
    # 合并特征和目标变量
    train_data = X_train_scaled.copy()
    train_data['median_house_value'] = y_train.values
    
    test_data = X_test_scaled.copy()
    test_data['median_house_value'] = y_test.values
    
    # 保存处理后的数据
    train_data.to_csv('data/processed/train.csv', index=False)
    test_data.to_csv('data/processed/test.csv', index=False)
    
    # 保存预处理对象
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('data/processed/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # 保存特征列信息
    with open('data/processed/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print(f"\n训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print("数据预处理完成！")
    
    # 显示处理后的数据信息
    print("\n处理后的训练数据前5行:")
    print(train_data.head())

if __name__ == "__main__":
    main()