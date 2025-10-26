# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 加载数据
print("正在加载加州房价数据集...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"数据集形状: {X.shape}")
print(f"特征名称: {housing.feature_names}")
print(f"目标变量范围: ${y.min():.2f} - ${y.max():.2f}万")

# 2. 数据探索
print("\n=== 数据探索 ===")
print(X.head())
print(f"\n数据基本信息:")
print(X.info())
print(f"\n数据描述性统计:")
print(X.describe())

# 3. 数据可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 房价分布
axes[0,0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_xlabel('房价（万美元）')
axes[0,0].set_ylabel('频数')
axes[0,0].set_title('房价分布')

# 特征相关性热力图
correlation_matrix = pd.concat([X, pd.Series(y, name='Price')], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0,1])
axes[0,1].set_title('特征相关性热力图')

# 最重要的特征与房价的关系
most_important_feature = correlation_matrix['Price'].abs().sort_values(ascending=False).index[1]
axes[1,0].scatter(X[most_important_feature], y, alpha=0.5)
axes[1,0].set_xlabel(most_important_feature)
axes[1,0].set_ylabel('房价（万美元）')
axes[1,0].set_title(f'{most_important_feature} vs 房价')

# 房间数与房价的关系
axes[1,1].scatter(X['AveRooms'], y, alpha=0.5, color='green')
axes[1,1].set_xlabel('平均房间数')
axes[1,1].set_ylabel('房价（万美元）')
axes[1,1].set_title('房间数与房价关系')

plt.tight_layout()
plt.show()

# 4. 数据预处理
print("\n=== 数据预处理 ===")
# 检查缺失值
print(f"缺失值数量: {X.isnull().sum().sum()}")

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 5. 模型训练和评估
print("\n=== 模型训练和评估 ===")

models = {
    '线性回归': LinearRegression(),
    '岭回归': Ridge(alpha=1.0),
    'Lasso回归': Lasso(alpha=0.1),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
    '支持向量机': SVR(kernel='rbf')
}

results = {}

for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")

# 6. 模型比较
print("\n=== 模型比较 ===")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'MAE': [results[name]['mae'] for name in results.keys()],
    'R2': [results[name]['r2'] for name in results.keys()]
}).sort_values('RMSE')

print(comparison_df)

# 可视化模型性能比较
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# RMSE比较
models_list = comparison_df['Model']
rmse_scores = comparison_df['RMSE']
bars = axes[0].bar(models_list, rmse_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet', 'orange'])
axes[0].set_title('模型RMSE比较')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=45)

# 在柱状图上添加数值
for bar, score in zip(bars, rmse_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')

# R²比较
r2_scores = comparison_df['R2']
bars = axes[1].bar(models_list, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet', 'orange'])
axes[1].set_title('模型R²分数比较')
axes[1].set_ylabel('R² Score')
axes[1].tick_params(axis='x', rotation=45)

# 在柱状图上添加数值
for bar, score in zip(bars, r2_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 7. 选择最佳模型并进行详细分析
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n最佳模型: {best_model_name}")

# 使用最佳模型进行预测
y_pred_best = best_model.predict(X_test)

# 预测结果可视化
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 实际值 vs 预测值散点图
axes[0].scatter(y_test, y_pred_best, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('实际房价')
axes[0].set_ylabel('预测房价')
axes[0].set_title(f'{best_model_name} - 实际值 vs 预测值')

# 残差图
residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('预测房价')
axes[1].set_ylabel('残差')
axes[1].set_title(f'{best_model_name} - 残差图')

plt.tight_layout()
plt.show()

# 8. 特征重要性分析（如果模型支持）
if hasattr(best_model, 'feature_importances_'):
    print("\n=== 特征重要性分析 ===")
    feature_importance = pd.DataFrame({
        'feature': housing.feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'{best_model_name} - 特征重要性')
    plt.tight_layout()
    plt.show()

# 9. 模型部署示例
print("\n=== 模型部署示例 ===")
print("使用最佳模型进行新数据预测:")

# 创建示例新数据（基于原始数据的统计信息）
new_data_example = np.array([[3.0, 20.0, 5.0, 1000.0, 500.0, 1000.0, 35.0, -120.0]])
new_data_scaled = scaler.transform(new_data_example)
predicted_price = best_model.predict(new_data_scaled)[0]

print(f"示例房屋特征:")
for i, feature in enumerate(housing.feature_names):
    print(f"  {feature}: {new_data_example[0][i]}")
print(f"预测房价: ${predicted_price:.2f}万")

# 10. 交叉验证
print("\n=== 交叉验证结果 ===")
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
print(f"5折交叉验证R²分数: {cv_scores}")
print(f"平均交叉验证R²分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n=== 房价预测模型训练完成 ===")
print("总结:")
print(f"1. 最佳模型: {best_model_name}")
print(f"2. 测试集R²分数: {results[best_model_name]['r2']:.4f}")
print(f"3. 测试集RMSE: {results[best_model_name]['rmse']:.4f}")
print(f"4. 平均绝对误差: {results[best_model_name]['mae']:.4f}")