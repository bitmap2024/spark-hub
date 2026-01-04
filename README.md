# 🔥 Spark Hub

> 一站式学术知识管理与发现平台 - 让知识触手可及

Spark Hub 是一个现代化的学术知识平台，集成了论文管理、知识库构建、智能推荐和社交互动等功能，帮助研究者和学习者更高效地获取、组织和分享学术知识。

## ✨ 功能特性

### 📚 知识管理
- **知识库创建** - 创建个人知识库，分类管理论文和笔记
- **论文收藏** - 收集、整理感兴趣的学术论文
- **标签系统** - 灵活的标签体系，快速检索内容

### 🤖 智能功能
- **智能推荐** - 基于深度学习的个性化推荐系统
- **论文解析** - 自动提取论文关键信息
- **搜索代理** - 集成 arXiv、Google Scholar 等学术搜索

### 👥 社交互动
- **关注系统** - 关注感兴趣的研究者
- **私信功能** - 与其他用户私密交流
- **内容分享** - 分享知识库和论文


## 🏗️ 项目结构

```
spark-hub/
├── spark-hub-frontend/     # 前端应用 (React + TypeScript + Vite)
├── spark-hub-backend/      # 后端服务 (FastAPI + PostgreSQL)
├── spark-hub-algorithm/    # 算法模块 (推荐系统、论文解析、搜索)
└── spark-hub-crawler/      # 数据爬虫 (学术论文采集)
```

## 🛠️ 技术栈


### 前端
- **React 18** - 用户界面框架
- **TypeScript** - 类型安全
- **Vite** - 快速构建工具
- **Tailwind CSS** - 原子化 CSS 框架
- **Shadcn UI** - 精美组件库
- **React Query** - 数据获取与缓存


### 爬虫
- **Scrapy** - 爬虫框架
- **反爬策略** - 代理池、请求调度
- **数据清洗** - Pipeline 处理

### 后端
- **FastAPI** - 高性能 Python Web 框架
- **SQLAlchemy** - ORM 框架
- **PostgreSQL** - 关系型数据库
- **JWT** - 身份认证
- **Docker** - 容器化部署

### 算法
- **推荐系统** - DeepFM, DIN, DIEN, Wide&Deep, 多目标推荐
- **搜索排序** - Bi-Encoder, ColBERT, Cross-Encoder Reranker
- **论文解析** - PDF 解析、信息抽取
- **LLM 集成** - 大语言模型服务



## 🚀 快速开始

### 前提条件

- Node.js 16+
- Python 3.9+
- PostgreSQL 13+

### 1. 克隆项目

```bash
git clone https://github.com/bitmap2024/spark-hub.git
cd spark-hub
```

### 2. 启动后端

```bash
cd spark-hub-backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置数据库连接等

# 初始化数据库
python init_db.py

# 启动服务
python serve.py
```

后端服务将运行在 http://localhost:8000

### 3. 启动前端

```bash
cd spark-hub-frontend

# 安装依赖
npm install

# 配置环境变量
echo "VITE_USE_MOCK_DATA=false" > .env
echo "VITE_API_BASE_URL=http://localhost:8000/api" >> .env

# 启动开发服务器
npm run dev
```

前端应用将运行在 http://localhost:5173

## 📖 API 文档

启动后端后，访问自动生成的 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Docker 部署

```bash
# 构建并启动所有服务
docker-compose up -d
```

## 📁 模块说明

### spark-hub-frontend
前端单页应用，提供用户界面和交互体验。

### spark-hub-backend
RESTful API 服务，处理业务逻辑和数据持久化。

### spark-hub-algorithm
算法服务模块：
- `recommentation_system/` - 推荐算法实现
- `spark_search_agent/` - 搜索与排序
- `paper_parse/` - 论文解析
- `saas-llm/` - LLM 服务集成

### spark-hub-crawler
数据采集模块，支持多源论文爬取。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📧 联系方式

- **GitHub**: [@bitmap2024](https://github.com/bitmap2024)
- **项目地址**: https://github.com/bitmap2024/spark-hub

---

<p align="center">
  Made with ❤️ by Spark Hub Team
</p>

