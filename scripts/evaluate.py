import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data():
    """加载测试数据"""
    test_data = pd.read_csv('data/processed/test.csv')
    X_test = test_data.drop('MedHouseVal', axis=1)
    y_test = test_data['MedHouseVal']
    return X_test, y_test

def load_model():
    """加载训练好的模型"""
    with open('models/best_model.pkl', 'rb') as f:
        return pickle.load(f)

def plot_predictions(y_true, y_pred, model_type):
    """绘制预测结果图"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 实际值 vs 预测值
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('实际房价')
    axes[0].set_ylabel('预测房价')
    axes[0].set_title(f'{model_type} - 实际值 vs 预测值')
    
    # 残差图
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('预测房价')
    axes[1].set_ylabel('残差')
    axes[1].set_title(f'{model_type} - 残差图')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/evaluation_plots.png')
    plt.close()

def main():
    print("开始模型评估...")
    
    # 加载数据和模型
    X_test, y_test = load_data()
    model = load_model()
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    
    # 记录到 MLflow
    with mlflow.start_run():
        mlflow.log_metrics({
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
    
    # 保存指标到 JSON 文件
    metrics = {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 绘制评估图表
    plot_predictions(y_test, y_pred, type(model).__name__)
    
    print("评估完成!")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")

if __name__ == "__main__":
    main()