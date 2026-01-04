# Spark Hub 后端

这是 Spark Hub 项目的后端 API 服务，提供用户管理、知识库管理、论文收集和私信等功能。

## 技术栈

- **FastAPI**: 现代化、高性能的 Python Web 框架
- **SQLAlchemy**: Python SQL 工具包和 ORM
- **Pydantic**: 数据验证和设置管理
- **PostgreSQL**: 关系型数据库
- **JWT**: 用于用户认证

## 功能特性

- 用户管理（注册、登录、个人资料）
- 社交关系（关注、粉丝）
- 知识库管理（创建、编辑、删除）
- 论文收集和组织
- 标签系统
- 私信功能

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 环境配置

创建 `.env` 文件，配置环境变量：

```
DATABASE_URL=postgresql://postgres:postgres@localhost/sparkhub
SECRET_KEY=your-secret-key
FIRST_SUPERUSER_EMAIL=admin@sparkhub.com
FIRST_SUPERUSER_PASSWORD=admin
FIRST_SUPERUSER_USERNAME=admin
```

### 3. 初始化数据库

```bash
python init_db.py
```

### 4. 运行服务

```bash
python serve.py
```

或者使用 uvicorn 直接运行：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

服务将在 http://localhost:8000 上运行。

## API 文档

启动服务后，可以访问自动生成的 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

### 认证

- `POST /api/v1/auth/login`: 用户登录
- `POST /api/v1/auth/signup`: 用户注册

### 用户

- `GET /api/v1/users/`: 获取用户列表
- `GET /api/v1/users/{user_id}`: 获取用户信息
- `GET /api/v1/users/username/{username}`: 通过用户名获取用户信息
- `PUT /api/v1/users/me`: 更新当前用户信息
- `POST /api/v1/users/follow/{user_id}`: 关注用户
- `POST /api/v1/users/unfollow/{user_id}`: 取消关注用户
- `GET /api/v1/users/me/following`: 获取当前用户关注的用户列表
- `GET /api/v1/users/me/followers`: 获取当前用户的粉丝列表

### 知识库

- `GET /api/v1/knowledge-bases/`: 获取所有知识库
- `POST /api/v1/knowledge-bases/`: 创建新知识库
- `GET /api/v1/knowledge-bases/{kb_id}`: 获取特定知识库
- `PUT /api/v1/knowledge-bases/{kb_id}`: 更新知识库
- `DELETE /api/v1/knowledge-bases/{kb_id}`: 删除知识库
- `POST /api/v1/knowledge-bases/{kb_id}/papers`: 添加论文到知识库
- `GET /api/v1/knowledge-bases/user/{user_id}`: 获取用户的知识库

### 论文

- `GET /api/v1/papers/{paper_id}`: 获取论文详情
- `PUT /api/v1/papers/{paper_id}`: 更新论文
- `DELETE /api/v1/papers/{paper_id}`: 删除论文
- `POST /api/v1/papers/{paper_id}/like`: 点赞论文
- `POST /api/v1/papers/{paper_id}/unlike`: 取消点赞论文
- `GET /api/v1/papers/liked`: 获取当前用户点赞的论文

### 标签

- `GET /api/v1/tags/`: 获取所有标签
- `POST /api/v1/tags/`: 创建新标签
- `GET /api/v1/tags/{tag_id}`: 获取标签详情
- `GET /api/v1/tags/{tag_id}/knowledge-bases`: 获取标签下的知识库
- `GET /api/v1/tags/{tag_id}/papers`: 获取标签下的论文

### 消息

- `GET /api/v1/messages/conversations`: 获取当前用户的所有会话
- `GET /api/v1/messages/{user_id}`: 获取与特定用户的消息历史
- `POST /api/v1/messages/`: 发送私信
- `POST /api/v1/messages/{message_id}/read`: 标记消息为已读
- `POST /api/v1/messages/{user_id}/read-all`: 标记与特定用户的所有消息为已读

## 项目结构

```
app/
├── api/                # API路由
│   ├── api_v1/
│   │   ├── endpoints/  # API端点
│   │   └── api.py      # API路由注册
│   └── deps.py         # 依赖项
├── core/               # 核心配置
│   ├── config.py       # 应用配置
│   └── security.py     # 安全工具
├── crud/               # 数据库CRUD操作
├── db/                 # 数据库
│   ├── session.py      # 数据库会话
│   └── init_db.py      # 数据库初始化
├── models/             # SQLAlchemy数据库模型
├── schemas/            # Pydantic模型
└── main.py             # 应用入口

tests/                  # 测试
requirements.txt        # 依赖项
serve.py                # 运行脚本
README.md               # 项目文档
```

## 开发

### 运行测试

```bash
pytest
``` 