# scripts/train_simple.py - 简化版训练脚本
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

def main():
    print("开始训练模型（简化版）...")
    
    # 加载数据
    train_data = pd.read_csv('data/processed/train.csv')
    X_train = train_data.drop('median_house_value', axis=1)
    y_train = train_data['median_house_value']
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    train_r2 = r2_score(y_train, y_pred)
    train_mae = mean_absolute_error(y_train, y_pred)
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"训练完成!")
    print(f"训练集 RMSE: {train_rmse:.4f}")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"训练集 MAE: {train_mae:.4f}")
    print("模型已保存到: models/best_model.pkl")

if __name__ == "__main__":
    main()