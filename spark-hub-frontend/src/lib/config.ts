// 环境配置
export const config = {
  // 是否使用模拟数据 - 设置为true表示所有API接口使用前端模拟数据，不依赖后端
  useMockData: true,
  
  // API 基础 URL - 仅在useMockData为false时使用
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  
  // 其他配置项
  appName: 'SparkHub',
  version: '1.0.0',
}; 