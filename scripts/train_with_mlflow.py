import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle

# 设置MLflow跟踪URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 开始实验
mlflow.set_experiment("房价预测")

def main():
    # 读取训练数据
    train_data = pd.read_csv('data/processed/train.csv')
    X_train = train_data.drop('median_house_value', axis=1)
    y_train = train_data['median_house_value']
    
    with mlflow.start_run():
        # 设置参数
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 记录模型
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # 记录评估指标（这里需要您添加评估逻辑）
        # 例如：
        # mlflow.log_metric("rmse", 53845.53)
        # mlflow.log_metric("r2", 0.7787)
        
        print("MLflow运行完成！访问 http://localhost:5000 查看结果")

if __name__ == "__main__":
    main()