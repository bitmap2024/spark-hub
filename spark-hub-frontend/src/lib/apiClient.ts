import { config } from './config';

// API 响应类型
export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
}

// API 客户端类
class ApiClient {
  private baseUrl: string;
  
  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }
  
  // 获取完整 URL
  private getFullUrl(endpoint: string): string {
    return `${this.baseUrl}${endpoint}`;
  }
  
  // 通用请求方法
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = this.getFullUrl(endpoint);
    
    // 添加默认请求头
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    
    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return { data };
    } catch (error) {
      console.error('API request failed:', error);
      return {
        data: {} as T,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
  
  // GET 请求
  async get<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'GET',
    });
  }
  
  // POST 请求
  async post<T>(endpoint: string, body: any, options: RequestInit = {}): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: JSON.stringify(body),
    });
  }
  
  // PUT 请求
  async put<T>(endpoint: string, body: any, options: RequestInit = {}): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: JSON.stringify(body),
    });
  }
  
  // DELETE 请求
  async delete<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'DELETE',
    });
  }
}

// 创建 API 客户端实例
export const apiClient = new ApiClient(config.apiBaseUrl); 