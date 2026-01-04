# Spark Hub Kubernetes部署指南

本文档提供了在Kubernetes环境中部署Spark Hub后端的详细指南。

## 部署架构

使用Kubernetes部署Spark Hub后端具有以下优势：

1. **容器化管理** - 通过容器化应用，实现环境一致性和隔离性
2. **自动扩展** - 根据负载自动水平扩展
3. **自动恢复** - 容器故障自动重启
4. **滚动更新** - 零停机升级
5. **声明式配置** - 基础设施即代码
6. **健康检查** - 活性和就绪探针确保服务健康

## 前置需求

- 运行中的Kubernetes集群
- kubectl命令行工具
- Docker和容器仓库访问权限

## 配置文件说明

- `deployment.yaml` - 定义应用部署
- `service.yaml` - 创建服务端点
- `configmap.yaml` - 配置非敏感参数
- `secret.yaml` - 存储敏感信息
- `pvc.yaml` - 持久卷声明用于日志存储
- `ingress.yaml` - 配置外部访问
- `hpa.yaml` - 水平Pod自动扩缩器
- `kustomization.yaml` - Kustomize配置文件

## 部署步骤

### 1. 构建并推送Docker镜像

```bash
# 在项目根目录下构建镜像
docker build -t your-registry/spark-hub-backend:latest .

# 推送镜像到镜像仓库
docker push your-registry/spark-hub-backend:latest
```

### 2. 修改配置

根据实际环境修改以下配置文件：

- `configmap.yaml` - 更新数据库连接信息
- `secret.yaml` - 使用正确的base64编码后的凭据
- `ingress.yaml` - 设置正确的域名
- `deployment.yaml` - 更新镜像地址和资源需求

### 3. 应用配置

```bash
# 创建命名空间（如果尚不存在）
kubectl create namespace spark-hub

# 使用Kustomize应用所有配置
kubectl apply -k .
```

### 4. 验证部署

```bash
# 检查部署状态
kubectl get deployments -n spark-hub

# 检查Pod状态
kubectl get pods -n spark-hub

# 检查服务
kubectl get services -n spark-hub

# 查看日志
kubectl logs deployment/spark-hub-backend -n spark-hub
```

## 开发环境配置

对于开发环境，我们保留了reload=true配置，以便支持代码更改后的自动重启。在`deployment.yaml`中，设置了环境变量：

```yaml
- name: RELOAD
  value: "true"
```

## 生产环境注意事项

1. **资源配置** - 根据实际负载调整CPU和内存需求
2. **副本数量** - 通过HPA自动调整，也可手动设置
3. **数据持久化** - 使用合适的存储类
4. **安全配置** - HTTPS配置、网络策略等
5. **监控** - 配置Prometheus和Grafana进行监控
6. **备份策略** - 定期备份数据库和配置

## 故障排除

### 常见问题

1. **Pod启动失败**
   - 检查镜像是否正确
   - 查看Pod日志：`kubectl logs <pod-name> -n spark-hub`

2. **数据库连接问题**
   - 验证Secret和ConfigMap配置
   - 确保数据库服务可访问

3. **健康检查失败**
   - 检查应用健康检查端点是否工作
   - 调整探针的超时和重试参数 