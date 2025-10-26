import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pickle
import os
import json

def load_params():
    """加载参数文件"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """加载处理后的数据"""
    train_data = pd.read_csv('data/processed/train.csv')
    X_train = train_data.drop('MedHouseVal', axis=1)
    y_train = train_data['MedHouseVal']
    return X_train, y_train

def get_model(model_type, model_params):
    """根据类型获取模型"""
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            random_state=42
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=42
        )
    elif model_type == "linear":
        return Ridge(alpha=model_params.get('alpha', 1.0))
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

def plot_feature_importance(model, feature_names, model_type):
    """绘制特征重要性图"""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title(f'{model_type} - 特征重要性')
        plt.tight_layout()
        
        # 创建目录并保存图片
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/feature_importance.png')
        plt.close()
        
        return importance
    return None

def main():
    # 加载参数
    params = load_params()
    train_params = params['train']
    model_type = train_params['model_type']
    model_params = params['models'].get(model_type, {})
    
    print(f"开始训练 {model_type} 模型...")
    
    # 设置 MLflow
    mlflow.set_experiment("house-price-prediction")
    
    with mlflow.start_run():
        # 记录参数
        mlflow.log_params({
            'model_type': model_type,
            **model_params
        })
        
        # 加载数据
        X_train, y_train = load_data()
        
        # 训练模型
        model = get_model(model_type, model_params)
        model.fit(X_train, y_train)
        
        # 在训练集上评估
        y_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_r2 = r2_score(y_train, y_pred)
        train_mae = mean_absolute_error(y_train, y_pred)
        
        # 记录指标
        mlflow.log_metrics({
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'train_mae': train_mae
        })
        
        # 记录模型
        mlflow.sklearn.log_model(model, "model")
        
        # 保存模型到本地
        os.makedirs('models', exist_ok=True)
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # 绘制特征重要性
        feature_importance = plot_feature_importance(model, X_train.columns, model_type)
        if feature_importance is not None:
            mlflow.log_artifact('models/feature_importance.png')
        
        print(f"训练完成!")
        print(f"训练集 RMSE: {train_rmse:.4f}")
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"训练集 MAE: {train_mae:.4f}")

if __name__ == "__main__":
    main()