# SparkHub 前端

这是 SparkHub 项目的前端部分，使用 React、TypeScript 和 Vite 构建。

## 功能特点

- 用户认证（登录/注册）
- 用户个人资料
- 关注/取消关注用户
- 知识库管理
- 私信系统
- 响应式设计

## 技术栈

- React 18
- TypeScript
- Vite
- React Router
- React Query
- Tailwind CSS
- Shadcn UI

## 开发环境设置

### 前提条件

- Node.js 16+
- npm 或 yarn

### 安装依赖

```bash
npm install
# 或
yarn
```

### 环境变量配置

创建 `.env` 文件（基于 `.env.example`）：

```
# 是否使用模拟数据（true 表示使用模拟数据，false 表示使用真实 API）
VITE_USE_MOCK_DATA=true

# API 基础 URL
VITE_API_BASE_URL=http://localhost:8000/api
```

### 启动开发服务器

```bash
npm run dev
# 或
yarn dev
```

## 构建生产版本

```bash
npm run build
# 或
yarn build
```

## 页面导航

应用的页面流程如下：

1. **个人主页**（第一个页面）：展示自己的知识库列表
2. **知识库详情页**（第二个页面）：点击知识库后进入，查看知识库的详细内容

## 项目结构

```
src/
├── components/     # 可复用组件
├── lib/            # 工具函数和 API 客户端
├── pages/          # 页面组件
├── hooks/          # 自定义 React Hooks
├── App.tsx         # 主应用组件
└── main.tsx        # 入口文件
```

## 前后端分离

本项目支持完全的前后端分离：

1. 当 `VITE_USE_MOCK_DATA=true` 时，前端使用模拟数据，不依赖后端服务
2. 当 `VITE_USE_MOCK_DATA=false` 时，前端通过 API 与后端通信

这使得前端开发人员可以在后端服务未就绪的情况下进行开发和测试。

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request
