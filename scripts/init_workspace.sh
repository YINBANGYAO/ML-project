#!/bin/bash

# 初始化工作空间脚本
echo "初始化房价预测项目工作空间..."

# 创建必要的目录
mkdir -p data/raw data/processed models plots notebooks scripts

# 复制环境变量文件（如果不存在）
if [ ! -f .env ]; then
    cp .env.example .env
    echo "请编辑 .env 文件配置您的环境变量"
fi

# 初始化 Git (如果尚未初始化)
if [ ! -d .git ]; then
    echo "初始化 Git 仓库..."
    git init
    git add .
    git commit -m "Initial commit: House Price Prediction Project"
fi

# 初始化 DVC (如果尚未初始化)
if [ ! -d .dvc ]; then
    echo "初始化 DVC..."
    dvc init
    git add .dvc
    git commit -m "Initialize DVC"
fi

echo "工作空间初始化完成!"
echo "接下来您可以:"
echo "1. 编辑 .env 文件配置环境变量"
echo "2. 运行 'docker-compose up -d' 启动服务"
echo "3. 访问 http://localhost:5000 查看 MLflow"
echo "4. 访问 http://localhost:8888 使用 Jupyter Lab"