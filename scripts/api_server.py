# scripts/api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
import uvicorn
from typing import List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="房价预测API", 
    version="1.0.0",
    description="基于机器学习的房价预测服务"
)

# 定义请求模型
class HouseFeatures(BaseModel):
    MedInc: float = 8.3252
    HouseAge: float = 41.0
    AveRooms: float = 6.9841
    AveBedrms: float = 1.0238
    Population: float = 322.0
    AveOccup: float = 2.5556
    Latitude: float = 37.88
    Longitude: float = -122.23

class PredictionRequest(BaseModel):
    features: List[HouseFeatures]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    status: str

# 加载模型
def load_model():
    """加载训练好的模型"""
    try:
        model_path = "models/best_model.pkl"
        if os.path.exists(model_path):
            logger.info(f"加载模型: {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"模型加载成功: {type(model).__name__}")
            return model
        else:
            logger.warning(f"模型文件不存在: {model_path}")
            return None
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None

model = load_model()

@app.get("/")
async def root():
    return {
        "message": "房价预测API", 
        "status": "运行中",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="模型未加载，服务不可用")
        
        # 转换输入数据
        features_list = [item.dict() for item in request.features]
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"收到预测请求，样本数: {len(features_list)}")
        
        # 进行预测
        predictions = model.predict(features_df)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_version="1.0.0",
            status="success"
        )
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    model_type = type(model).__name__
    if hasattr(model, 'feature_importances_'):
        feature_count = len(model.feature_importances_)
    else:
        feature_count = "unknown"
    
    return {
        "model_type": model_type,
        "feature_count": feature_count,
        "features": ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                    'Population', 'AveOccup', 'Latitude', 'Longitude'],
        "status": "loaded"
    }

@app.get("/debug/containers")
async def debug_containers():
    """调试端点：检查容器内文件"""
    import subprocess
    try:
        # 检查关键文件
        files = {}
        check_paths = [
            "models/",
            "models/best_model.pkl",
            "scripts/api_server.py",
            "data/processed/train.csv"
        ]
        
        for path in check_paths:
            files[path] = os.path.exists(path)
        
        # 检查进程
        processes = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        return {
            "files": files,
            "processes": processes.stdout.split('\n')[:10],  # 前10个进程
            "working_directory": os.getcwd(),
            "python_path": os.environ.get('PYTHONPATH', '未设置')
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("启动房价预测API服务...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )