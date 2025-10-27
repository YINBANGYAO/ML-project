import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

# 添加 scripts 目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

class TestTraining:
    
    def test_model_creation(self):
        """测试模型创建"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        
        # 测试随机森林
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        assert rf_model.n_estimators == 10
        
        # 测试梯度提升
        gb_model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, random_state=42)
        assert gb_model.n_estimators == 10
        assert gb_model.learning_rate == 0.1
        
        # 测试岭回归
        ridge_model = Ridge(alpha=1.0)
        assert ridge_model.alpha == 1.0
        print("✓ 模型创建测试通过")
    
    def test_model_training(self, processed_data):
        """测试模型训练"""
        from sklearn.ensemble import RandomForestRegressor
        
        X, y = processed_data
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # 验证模型已训练
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
        assert hasattr(model, 'predict')
        print("✓ 模型训练测试通过")
    
    def test_model_prediction(self, processed_data):
        """测试模型预测"""
        from sklearn.ensemble import RandomForestRegressor
        
        X, y = processed_data
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # 进行预测
        predictions = model.predict(X.head(10))
        
        # 验证预测结果
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype in [np.float32, np.float64]
        print("✓ 模型预测测试通过")
    
    def test_model_evaluation(self, processed_data):
        """测试模型评估"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        X, y = processed_data
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 进行预测
        predictions = model.predict(X_test)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        # 验证指标
        assert rmse >= 0
        assert -1 <= r2 <= 1
        assert mae >= 0
        print("✓ 模型评估测试通过")
    
    def test_model_persistence(self, processed_data, temp_dir):
        """测试模型保存和加载"""
        from sklearn.ensemble import RandomForestRegressor
        
        X, y = processed_data
        
        # 创建简单模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # 保存模型
        model_path = Path(temp_dir) / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 验证模型文件存在
        assert model_path.exists()
        
        # 加载模型
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # 验证加载的模型
        assert isinstance(loaded_model, RandomForestRegressor)
        assert hasattr(loaded_model, 'predict')
        
        # 验证预测一致性
        original_predictions = model.predict(X.head(5))
        loaded_predictions = loaded_model.predict(X.head(5))
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
        print("✓ 模型持久化测试通过")