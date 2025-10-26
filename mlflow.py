import mlflow
import subprocess
import webbrowser
import os

def start_mlflow_ui():
    """启动 MLflow UI"""
    print("启动 MLflow UI...")
    # 在后台启动 MLflow UI
    subprocess.Popen(["mlflow", "ui", "--port", "5000"])
    print("MLflow UI 已启动: http://localhost:5000")
    webbrowser.open("http://localhost:5000")

def compare_runs(experiment_name="house-price-prediction"):
    """比较不同实验运行"""
    mlflow.set_experiment(experiment_name)
    runs = mlflow.search_runs()
    
    print("实验运行比较:")
    print(runs[['run_id', 'params.model_type', 'metrics.test_rmse', 'metrics.test_r2']].head())

if __name__ == "__main__":
    start_mlflow_ui()