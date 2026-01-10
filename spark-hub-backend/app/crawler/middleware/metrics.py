from prometheus_client import Counter, Gauge, Histogram, start_http_server
from typing import Dict, Optional
from loguru import logger
import time
from config.settings import PROMETHEUS_CONFIG

class CrawlerMetrics:
    def __init__(self, port: int = PROMETHEUS_CONFIG["port"]):
        # 请求计数器
        self.requests_total = Counter(
            "crawler_requests_total",
            "Total number of requests made",
            ["status", "domain"]
        )
        
        # 请求延迟直方图
        self.request_duration_seconds = Histogram(
            "crawler_request_duration_seconds",
            "Request duration in seconds",
            ["domain"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
        )
        
        # 当前活跃请求数
        self.active_requests = Gauge(
            "crawler_active_requests",
            "Number of requests currently being processed",
            ["domain"]
        )
        
        # URL队列大小
        self.queue_size = Gauge(
            "crawler_queue_size",
            "Number of URLs in different queues",
            ["queue_type"]
        )
        
        # 代理池状态
        self.proxy_pool_size = Gauge(
            "crawler_proxy_pool_size",
            "Number of proxies in the pool",
            ["status"]
        )
        
        # 内存使用
        self.memory_usage_bytes = Gauge(
            "crawler_memory_usage_bytes",
            "Memory usage in bytes"
        )
        
        # 存储统计
        self.storage_size_bytes = Gauge(
            "crawler_storage_size_bytes",
            "Storage size in bytes",
            ["storage_type"]
        )
        
        # 错误计数器
        self.errors_total = Counter(
            "crawler_errors_total",
            "Total number of errors",
            ["error_type"]
        )
        
        # 启动HTTP服务器
        start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")
        
    def track_request(self, domain: str):
        """
        跟踪单个请求的上下文管理器
        
        用法:
        async with metrics.track_request("example.com") as tracker:
            response = await make_request()
            tracker.set_status(response.status)
        """
        return RequestTracker(self, domain)
        
    def update_queue_size(self, queue_type: str, size: int):
        """更新队列大小"""
        self.queue_size.labels(queue_type=queue_type).set(size)
        
    def update_proxy_stats(self, good: int, bad: int):
        """更新代理池统计"""
        self.proxy_pool_size.labels(status="good").set(good)
        self.proxy_pool_size.labels(status="bad").set(bad)
        
    def update_storage_size(self, storage_type: str, size: int):
        """更新存储大小"""
        self.storage_size_bytes.labels(storage_type=storage_type).set(size)
        
    def record_error(self, error_type: str):
        """记录错误"""
        self.errors_total.labels(error_type=error_type).inc()

class RequestTracker:
    def __init__(self, metrics: CrawlerMetrics, domain: str):
        self.metrics = metrics
        self.domain = domain
        self.start_time = None
        self.status: Optional[int] = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        self.metrics.active_requests.labels(domain=self.domain).inc()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics.active_requests.labels(domain=self.domain).dec()
        
        # 记录请求时间
        self.metrics.request_duration_seconds.labels(
            domain=self.domain
        ).observe(duration)
        
        # 记录请求状态
        if self.status is not None:
            status_group = f"{self.status // 100}xx"
            self.metrics.requests_total.labels(
                status=status_group,
                domain=self.domain
            ).inc()
            
        # 如果有异常，记录错误
        if exc_type is not None:
            self.metrics.errors_total.labels(
                error_type=exc_type.__name__
            ).inc()
            
    def set_status(self, status: int):
        """设置请求状态码"""
        self.status = status 