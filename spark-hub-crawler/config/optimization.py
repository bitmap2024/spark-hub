"""
性能优化配置模块
"""
from typing import Dict, Any

# 并发与异步配置
CONCURRENCY_SETTINGS: Dict[str, Any] = {
    'max_concurrent_requests': 32,      # 最大并发请求数
    'max_concurrent_domains': 8,        # 每个域名最大并发
    'max_concurrent_pipelines': 16,     # 管道处理并发数
    'async_dns_lookup': True,           # 异步DNS解析
    'connection_pool_size': 100,        # 连接池大小
    'max_retry_times': 3               # 最大重试次数
}

# 异步协程配置
ASYNC_SETTINGS: Dict[str, Any] = {
    'event_loop_policy': 'uvloop',      # 使用uvloop替代默认事件循环
    'timeout_settings': {
        'dns_timeout': 5,
        'connect_timeout': 10,
        'read_timeout': 30
    }
}

# 内存与缓存配置
MEMORY_SETTINGS: Dict[str, Any] = {
    'item_buffer_size': 1000,           # 数据项缓冲区大小
    'response_compression': True,        # 响应体压缩
    'max_response_size': 10 * 1024*1024, # 最大响应大小限制
    'cache_settings': {
        'enable_cache': True,
        'cache_backend': 'redis',
        'cache_expiration': 3600,
        'max_cache_size': 1000000
    }
}

# 网络优化配置
NETWORK_SETTINGS: Dict[str, Any] = {
    'keep_alive': True,                 # 保持连接
    'tcp_nodelay': True,                # TCP_NODELAY
    'dns_cache': True,                  # DNS缓存
    'proxy_settings': {
        'enable_proxy': True,
        'proxy_pool_size': 100,         # 代理池大小
        'proxy_rotation_interval': 60,   # 代理轮换间隔(秒)
        'check_proxy_interval': 300     # 代理检查间隔(秒)
    }
}

# 分布式配置
DISTRIBUTED_SETTINGS: Dict[str, Any] = {
    'enable_distributed': True,
    'scheduler': {
        'type': 'redis',
        'batch_size': 1000,             # 任务批处理大小
        'queue_timeout': 5,             # 队列超时时间
    },
    'worker': {
        'prefetch_count': 100,          # 预取任务数
        'heartbeat_interval': 30,       # 心跳间隔
        'max_tasks_per_worker': 10000   # 单个工作进程最大任务数
    }
}

# 性能监控指标
METRICS: Dict[str, Any] = {
    'crawler_metrics': {
        'requests_per_second': True,     # 每秒请求数
        'success_rate': True,           # 成功率
        'average_response_time': True,  # 平均响应时间
        'bandwidth_usage': True         # 带宽使用
    },
    'system_metrics': {
        'cpu_usage': True,             # CPU使用率
        'memory_usage': True,          # 内存使用
        'disk_io': True,              # 磁盘IO
        'network_io': True            # 网络IO
    }
} 