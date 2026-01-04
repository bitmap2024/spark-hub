# Spark Hub 部署指南

本文档提供了在生产环境中部署Spark Hub后端的指南。

## 文件说明

- `gunicorn_conf.py`: Gunicorn配置文件
- `supervisor_conf.ini`: Supervisor配置文件
- `nginx_conf.conf`: Nginx配置文件
- `systemd_service.service`: systemd服务文件
- `k8s/`: Kubernetes部署配置文件

## 部署步骤

### Kubernetes部署（推荐用于生产环境）

如果使用Kubernetes进行部署，可以忽略传统的Supervisor、systemd和Nginx配置，直接使用以下Kubernetes配置：

```bash
# 应用Kubernetes配置
kubectl apply -f deploy/k8s/
```

对于Kubernetes部署，我们主要关注以下几点：
1. 容器镜像构建 - 利用多阶段构建优化镜像大小
2. 资源配置 - 根据实际需求配置CPU和内存资源
3. 健康检查 - 配置Liveness和Readiness探针
4. 配置管理 - 使用ConfigMap和Secret管理配置
5. 持久化存储 - 使用PVC管理需要持久化的数据
6. 水平扩展 - 配置HPA根据负载自动扩展

详见`deploy/k8s/`目录下的配置文件。

### 传统部署（适用于非Kubernetes环境）

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装生产环境额外依赖
pip install gunicorn
```

### 2. 使用Gunicorn运行应用

```bash
# 使用Gunicorn配置文件运行
gunicorn -c deploy/gunicorn_conf.py app.main:app
```

### 3. 使用Supervisor管理进程

```bash
# 编辑supervisor配置文件，替换其中的路径为实际路径
sudo cp deploy/supervisor_conf.ini /etc/supervisor/conf.d/spark-hub.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start spark-hub
```

### 4. 使用systemd管理服务

```bash
# 编辑systemd服务文件，替换其中的路径为实际路径
sudo cp deploy/systemd_service.service /etc/systemd/system/spark-hub.service
sudo systemctl daemon-reload
sudo systemctl enable spark-hub
sudo systemctl start spark-hub
```

### 5. 配置Nginx

```bash
# 编辑Nginx配置文件，替换域名和SSL证书路径
sudo cp deploy/nginx_conf.conf /etc/nginx/sites-available/spark-hub
sudo ln -s /etc/nginx/sites-available/spark-hub /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 监控与维护

### 检查服务状态

```bash
# 检查Supervisor状态
sudo supervisorctl status spark-hub

# 检查systemd服务状态
sudo systemctl status spark-hub

# 检查Nginx状态
sudo systemctl status nginx
```

### 查看日志

```bash
# 应用日志
tail -f logs/app.log

# 访问日志
tail -f logs/access.log

# 错误日志
tail -f logs/error.log
```

### 重启服务

```bash
# 使用Supervisor重启
sudo supervisorctl restart spark-hub

# 使用systemd重启
sudo systemctl restart spark-hub
```

## 注意事项

1. 所有配置文件中的路径需要根据实际部署环境进行修改
2. 在生产环境中，建议使用HTTPS并配置适当的SSL证书
3. 根据实际需求调整Gunicorn的worker数量
4. 定期检查日志并实施监控
5. 注意备份数据库和关键配置文件

## 部署选择总结

### Kubernetes部署（推荐用于生产环境）

**优势**：
- 水平扩展和自动伸缩
- 自动恢复故障
- 滚动更新和零停机部署
- 配置和密钥管理
- 自动化运维
- 资源利用率高

**适用场景**：
- 中大型应用
- 需要高可用性的服务
- 流量波动较大的应用
- 微服务架构

### 传统部署

**优势**：
- 部署架构简单
- 适合小型应用
- 运维成本较低
- 学习成本低

**适用场景**：
- 流量稳定的小型应用
- 单一服务
- 资源有限的环境
- 开发和测试环境

## 推荐配置

无论选择哪种部署方式，以下配置都是推荐的：

1. 使用环境变量进行配置
2. 实施日志轮转和管理
3. 实施监控和告警
4. 配置适当的健康检查
5. 定期备份数据
6. 遵循最小权限原则 