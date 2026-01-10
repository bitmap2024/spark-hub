# Spark Hub Crawler

高性能分布式爬虫系统，具有强大的反爬虫能力和智能资源管理。

## 中文网页统计

### 1. 规模概况
- **总体规模**：
  - 活跃网页：约150-200亿页
  - 含历史版本：约400-500亿页
  - 日均新增：约500-800万页

### 2. 分类分布
- **新闻媒体**：15-20%
  - 新闻网站
  - 媒体平台
  - 资讯门户
  
- **社交平台**：25-30%
  - 社交网络
  - 微博/微信
  - 短视频平台
  
- **电商平台**：20-25%
  - 综合电商
  - 垂直电商
  - 本地服务
  
- **博客/论坛**：15-20%
  - 博客平台
  - 问答社区
  - 专业论坛
  
- **政府/教育**：10-15%
  - 政府网站
  - 教育机构
  - 研究机构
  
- **其他**：5-10%
  - 企业官网
  - 个人网站
  - 其他服务

### 3. 更新频率
- **实时更新**：15%（新闻、社交）
- **每日更新**：30%（电商、资讯）
- **周期更新**：40%（博客、论坛）
- **低频更新**：15%（政府、企业）

## 爬取策略建议

### 1. 分级爬取
- **核心内容**：
  - 优先级：最高
  - 更新频率：小时级
  - 覆盖范围：10-15%
  
- **重要内容**：
  - 优先级：高
  - 更新频率：天级
  - 覆盖范围：20-25%
  
- **普通内容**：
  - 优先级：中
  - 更新频率：周级
  - 覆盖范围：40-50%
  
- **低频内容**：
  - 优先级：低
  - 更新频率：月级
  - 覆盖范围：15-20%

### 2. 存储需求
- **原始数据**：
  - 每页平均：50-100KB
  - 压缩比：3-5倍
  - 增量存储：建议启用
  
- **索引数据**：
  - 元数据索引：约原始数据的10%
  - 全文索引：约原始数据的30%
  - 实时索引：必要

### 3. 资源配置
- **存储系统**：
  - 分布式文件系统
  - 对象存储
  - 时序数据库
  
- **计算资源**：
  - 爬虫节点：根据规模配置
  - 解析节点：爬虫节点的50%
  - 索引节点：爬虫节点的30%

## 系统要求

### 基础环境
- Python 3.8+
- pip 21.0+
- gcc/clang 编译器
- OpenSSL 开发库
- Python 开发头文件

### 操作系统支持
- Linux (推荐 Ubuntu 20.04+)
- macOS (10.15+)
- Windows 10+ (需要额外配置)

## 安装指南

### 1. 准备环境

#### Ubuntu/Debian
```bash
# 安装编译工具和依赖
sudo apt-get update
sudo apt-get install -y python3-dev build-essential libssl-dev python3-dev

# 如果使用 Python 3.11
sudo apt-get install python3.11-dev
```

#### macOS
```bash
# 使用 Homebrew 安装依赖
brew install openssl

# 设置编译环境变量
export LDFLAGS="-L/usr/local/opt/openssl/lib"
export CPPFLAGS="-I/usr/local/opt/openssl/include"

# 如果使用 conda 环境
conda install python-dev
```

#### Windows
```bash
# 安装 Visual C++ 构建工具
# 下载并安装：https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 选择安装"Python 开发"和"C++ 构建工具"
```

### 2. 创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS
source venv/bin/activate
# Windows
.\venv\Scripts\activate
```

### 3. 安装依赖

#### 基础安装
```bash
# 更新 pip
pip install --upgrade pip setuptools wheel

# 安装依赖
pip install -r requirements.txt
```

#### 解决 cchardet 安装问题

如果遇到 `longintrepr.h` 相关错误，请尝试以下解决方案：

1. 使用预编译包（推荐）：
```bash
# 方案1：使用预编译的二进制包
pip install --only-binary :all: cchardet
```

2. 手动安装 Python 开发包：
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# macOS (使用 conda)
conda install python-dev

# macOS (使用系统 Python)
brew install python3
```

3. 使用替代方案：
```bash
# 方案2：使用替代包
pip install charset-normalizer

# 编辑 requirements.txt，将：
# cchardet==2.1.7
# 替换为：
# charset-normalizer>=2.0.0
```

4. 如果使用 conda 环境：
```bash
# 方案3：使用 conda 安装
conda install -c conda-forge cchardet
```

注意：如果以上方案都不能解决问题，请确保：
1. Python 开发环境正确安装
2. 编译器（gcc/clang）正确安装
3. 系统环境变量正确设置

## 系统特点

### 1. 性能优化
- **异步并发**：使用 `uvloop` 替代默认事件循环，提升异步性能
- **内存优化**：实现响应压缩和智能缓冲区管理
- **网络优化**：支持连接池和 DNS 缓存
- **批量处理**：支持批量请求和响应处理
- **监控指标**：详细的性能监控和统计

### 2. 爬取性能

#### 2.1 基准速率（单机）
- 静态页面：8-10万页面/小时
- 动态页面：3-4万页面/小时
- 代理模式：5-6万页面/小时

#### 2.2 持续运行性能
- 周运行时间：约142小时（考虑85%稳定性）
- 静态页面：900-1200万页面/周
- 动态页面：340-450万页面/周
- 混合模式：570-680万页面/周

#### 2.3 历史数据爬取
- 支持时间范围：可指定1-3年
- 支持增量更新
- 自动处理历史版本
- 智能去重和过滤

#### 2.4 性能优化建议
- 多区域部署：建议3-5个区域
- 代理池规模：建议300-500个有效代理
- 并发配置：
  - 静态页面：100-150线程
  - 动态页面：30-50线程
  - 代理模式：50-80线程
- 内存配置：建议16GB以上
- 存储配置：建议SSD，500GB以上

### 3. 反爬虫机制
- **User-Agent管理**
  - 智能UA轮换
  - 自定义UA池
  - 浏览器特征模拟

- **Cookie管理**
  - 域名级别Cookie池
  - 智能Cookie轮换
  - 自动Cookie提取

- **代理IP管理**
  - 代理IP评分系统
  - 智能代理轮换
  - 自动淘汰机制
  - 代理健康检查

- **请求限速**
  - 自适应限速算法
  - 域名级别限速
  - 智能请求队列
  - 动态调整策略

- **行为模拟**
  - 真实鼠标轨迹
  - 智能页面滚动
  - 随机点击行为
  - 键盘输入模拟

- **指纹管理**
  - 浏览器指纹池
  - WebGL指纹
  - Canvas指纹
  - 音频指纹

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 基本使用
```python
from spark_hub_crawler import SparkCrawler

# 创建爬虫实例
crawler = SparkCrawler()

# 配置爬虫
crawler.configure(
    max_concurrency=100,
    request_timeout=30,
    retry_times=3
)

# 启动爬取
await crawler.start()
```

### 3. 自定义配置
```python
# 配置代理
crawler.set_proxy_pool([
    "http://proxy1:8080",
    "http://proxy2:8080"
])

# 配置User-Agent
crawler.add_custom_ua([
    "Mozilla/5.0 ...",
    "Mozilla/5.0 ..."
])

# 设置域名限速
crawler.set_rate_limit("example.com", max_requests=10)
```

## 高级功能

### 1. 反爬虫配置
```python
# 配置反爬虫策略
crawler.configure_anti_crawler({
    'max_retries': 3,
    'retry_delay': 5,
    'success_codes': {200, 201, 202},
    'block_codes': {403, 429, 430}
})
```

### 2. 行为模拟
```python
# 配置行为模拟
crawler.configure_behavior({
    'scroll_probability': 0.7,
    'click_probability': 0.3,
    'typing_speed': (50, 200)
})
```

### 3. 监控指标
```python
# 获取爬虫状态
stats = crawler.get_metrics()
print(f"总请求数: {stats['total_requests']}")
print(f"成功率: {stats['success_rate']:.2%}")
```

## 性能监控

### 1. 指标说明
- **请求统计**
  - 总请求数
  - 成功请求数
  - 失败请求数
  - 被封禁请求数

- **响应时间**
  - 平均响应时间
  - 响应时间分布
  - 超时统计

- **资源使用**
  - 代理IP使用情况
  - Cookie池状态
  - UA池状态

### 2. 监控接口
```python
# 获取域名级别统计
domain_stats = crawler.get_metrics("example.com")

# 获取代理状态
proxy_stats = crawler.get_proxy_stats()

# 获取Cookie状态
cookie_stats = crawler.get_cookie_stats()
```

## 最佳实践

### 1. 性能优化
- 根据目标网站调整并发数
- 合理设置超时时间
- 适当配置重试策略
- 优化资源池大小

### 2. 反爬虫策略
- 根据网站特点选择合适的模拟策略
- 动态调整请求频率
- 及时更新代理池
- 保持Cookie新鲜度

### 3. 资源管理
- 定期清理无效代理
- 及时更新Cookie池
- 动态调整UA策略
- 监控系统资源使用

## 注意事项

1. 请遵守目标网站的robots.txt规则
2. 合理控制请求频率，避免对目标站点造成压力
3. 定期检查和更新代理池
4. 关注系统监控指标，及时调整策略

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 更新日志

### v1.0.0 (2024-03)
- 实现基础爬虫功能
- 添加反爬虫机制
- 实现性能优化
- 添加监控系统

## 超大规模爬虫优化

### 1. 架构升级
- **微服务化改造**
  - 爬虫引擎微服务
  - URL管理微服务
  - 代理池微服务
  - 解析引擎微服务
  - 存储服务微服务
  - 监控告警微服务

- **分布式协同**
  - Kubernetes集群管理
  - 服务网格(Service Mesh)
  - 分布式配置中心
  - 分布式任务调度

### 2. 性能突破（10-1000倍提升）

#### 2.1 硬件层面
- **规模化部署**
  - 全球50-100个区域
  - 每个区域100-1000个节点
  - 每个节点16-32核CPU
  - 每个节点64-128GB内存
  
- **网络优化**
  - 专线网络
  - CDN加速
  - 边缘节点
  - 智能DNS

#### 2.2 软件层面
- **并发优化**
  - 协程池：10000-50000/节点
  - 连接池：1000-5000/节点
  - 异步IO：uvloop加速
  - 零拷贝技术

- **内存优化**
  - 共享内存
  - 内存池
  - 对象复用
  - Jemalloc内存分配

#### 2.3 算法优化
- **智能调度**
  - 负载预测
  - 自适应限速
  - 动态资源分配
  - 任务优先级

- **数据处理**
  - 流式处理
  - 增量更新
  - 实时压缩
  - 智能过滤

### 3. 分布式架构

#### 3.1 核心组件
- **调度中心**
  - 全局任务分发
  - 负载均衡
  - 故障转移
  - 实时监控

- **数据中心**
  - 分布式存储
  - 数据分片
  - 副本管理
  - 数据同步

- **控制中心**
  - 配置管理
  - 服务编排
  - 资源调度
  - 监控告警

#### 3.2 存储优化
- **分层存储**
  - 内存缓存：Redis集群
  - 时序数据：InfluxDB
  - 对象存储：MinIO
  - 文档存储：Elasticsearch

- **索引优化**
  - 分片索引
  - 实时索引
  - 增量索引
  - 压缩索引

### 4. 智能化提升

#### 4.1 AI辅助
- **智能调度**
  - 流量预测
  - 资源预分配
  - 自动扩缩容
  - 异常检测

- **智能解析**
  - 自动提取
  - 模板学习
  - 结构识别
  - 内容分类

#### 4.2 自适应优化
- **自动调参**
  - 并发度
  - 超时时间
  - 重试策略
  - 批处理大小

- **智能代理**
  - 代理评分
  - 自动切换
  - 故障预测
  - 性能优化

### 5. 部署方案

#### 5.1 基础设施
- **计算资源**
  - AWS/阿里云/腾讯云
  - 自建IDC
  - 边缘节点
  - 专线网络

- **存储资源**
  - 分布式文件系统
  - 对象存储
  - 内存数据库
  - 时序数据库

#### 5.2 监控体系
- **性能监控**
  - QPS监控
  - 延迟监控
  - 资源使用
  - 错误率

- **业务监控**
  - 覆盖率
  - 成功率
  - 重复率
  - 质量评分

### 6. 成本估算

#### 6.1 硬件成本（月）
- **计算节点**：
  - 1000节点：约100-150万
  - 5000节点：约400-600万
  - 10000节点：约800-1200万

- **存储节点**：
  - 基础存储：约50-80万
  - 高速缓存：约30-50万
  - 备份存储：约20-30万

#### 6.2 带宽成本（月）
- **国内带宽**：
  - 基础：约30-50万
  - 高峰：约50-80万

- **国际带宽**：
  - 基础：约50-80万
  - 高峰：约80-120万
