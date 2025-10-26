#!/bin/bash

# 等待 MLflow 服务启动
echo "等待 MLflow 服务启动..."
while ! nc -z mlflow 5000; do
  sleep 1
done
echo "MLflow 服务已启动"

# 初始化 DVC (如果尚未初始化)
if [ ! -d .dvc ]; then
    echo "初始化 DVC..."
    dvc init --no-scm
fi

# 设置 DVC 远程存储 (如果提供了环境变量)
if [ ! -z "$DVC_REMOTE_URL" ] && [ ! -z "$DVC_REMOTE_NAME" ]; then
    echo "设置 DVC 远程存储: $DVC_REMOTE_URL"
    dvc remote add $DVC_REMOTE_NAME $DVC_REMOTE_URL
    dvc remote default $DVC_REMOTE_NAME
fi

# 执行传入的命令
exec "$@"