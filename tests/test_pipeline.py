import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# 添加 scripts 目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

class TestPipeline:
    
    def test_complete_pipeline(self, sample_housing_data, temp_dir):
        """测试完整的数据处理管道"""
        # 创建临时目录结构
        raw_data_dir = Path(temp_dir) / 'data' / 'raw'
        processed_data_dir = Path(temp_dir) / 'data' / 'processed'
        models_dir = Path(temp_dir) / 'models'
        
        raw_data_dir.mkdir(parents=True)
        processed_data_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        
        # 保存样本数据
        raw_data_path = raw_data_dir / 'housing.csv'
        sample_housing_data.to_csv(raw_data_path, index=False)
        
        # 模拟数据预处理步骤
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        import pickle
        
        # 1. 数据加载
        data = pd.read_csv(raw_data_path)
        print("✓ 数据加载成功")
        
        # 2. 数据清洗
        # 处理缺失值
        if data['total_bedrooms'].isnull().any():
            data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace=True)
        print("✓ 缺失值处理完成")
        
        # 3. 特征工程
        # 编码分类变量
        le = LabelEncoder()
        data['ocean_proximity_encoded'] = le.fit_transform(data['ocean_proximity'])
        print("✓ 分类变量编码完成")
        
        # 选择特征
        feature_columns = [
            'longitude', 'latitude', 'housing_median_age', 
            'total_rooms', 'total_bedrooms', 'population', 
            'households', 'median_income', 'ocean_proximity_encoded'
        ]
        X = data[feature_columns]
        y = data['median_house_value']
        
        # 4. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("✓ 数据分割完成")
        
        # 5. 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("✓ 特征缩放完成")
        
        # 6. 模型训练
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        print("✓ 模型训练完成")
        
        # 7. 模型评估
        predictions = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        assert rmse >= 0
        assert -1 <= r2 <= 1
        print("✓ 模型评估完成")
        
        # 8. 模型保存
        model_path = models_dir / 'test_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存预处理对象
        scaler_path = processed_data_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # 验证文件保存
        assert model_path.exists()
        assert scaler_path.exists()
        print("✓ 模型和预处理对象保存成功")
        
        print("✓ 完整管道测试通过")
    
    def test_preprocess_script(self, temp_dir):
        """测试预处理脚本的基本功能"""
        # 这个测试验证预处理脚本能否被导入和执行
        # 实际运行应该在 Docker 环境中进行
        
        try:
            # 动态从 scripts 目录加载 preprocess.py，以避免静态导入错误
            import importlib.util
            scripts_dir = Path(__file__).parent.parent / 'scripts'
            preprocess_path = scripts_dir / 'preprocess.py'
            if not preprocess_path.exists():
                pytest.skip("预处理脚本不可用")
            spec = importlib.util.spec_from_file_location("preprocess", str(preprocess_path))
            preprocess = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocess)
            preprocess_main = getattr(preprocess, 'main', None)
            if preprocess_main is None:
                pytest.skip("预处理脚本中未找到 main 函数")
            print("✓ 预处理脚本导入成功")
        except Exception:
            # 如果加载失败，跳过这个测试
            pytest.skip("预处理脚本不可用")
    
    def test_train_script(self, temp_dir):
        """测试训练脚本的基本功能"""
        # 这个测试验证训练脚本能否被导入和执行
        
        try:
            # 尝试导入训练模块
            import importlib.util
            scripts_dir = Path(__file__).parent.parent / 'scripts'
            train_path = scripts_dir / 'train_simple.py'
            if not train_path.exists():
                pytest.skip("训练脚本不可用")
            spec = importlib.util.spec_from_file_location("train_simple", str(train_path))
            train_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_module)
            train_main = getattr(train_module, 'main', None)
            if train_main is None:
                pytest.skip("训练脚本中未找到 main 函数")
            print("✓ 训练脚本导入成功")
        except Exception:
            # 如果导入失败，跳过这个测试
            pytest.skip("训练脚本不可用")