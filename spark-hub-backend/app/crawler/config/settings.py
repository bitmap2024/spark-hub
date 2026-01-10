from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 基础配置
BASE_DIR = Path(__file__).parent.parent
CONCURRENT_REQUESTS = 100  # 并发请求数
REQUEST_TIMEOUT = 30  # 请求超时时间
RETRY_TIMES = 3  # 重试次数
RETRY_DELAY = 3  # 重试延迟

# 请求配置
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# 代理配置
PROXY_ENABLED = True
PROXY_POOL_URL = os.getenv('PROXY_POOL_URL', '')

# 数据库配置
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'crawler')

MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'db': os.getenv('MYSQL_DB', 'crawler'),
}

# Redis配置
REDIS_URI = os.getenv('REDIS_URI', 'redis://localhost:6379')

# 日志配置
LOG_LEVEL = 'INFO'
LOG_FILE = BASE_DIR / 'logs' / 'crawler.log'

# Redis 配置
REDIS_CONFIG = {
    "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "queue_db": 0,
    "bloom_db": 1,
    "proxy_db": 2
}

# MongoDB 配置
MONGO_CONFIG = {
    "url": os.getenv("MONGO_URL", "mongodb://localhost:27017"),
    "db": "spark_crawler",
    "collections": {
        "pages": "raw_pages",
        "items": "parsed_items",
        "stats": "crawler_stats"
    }
}

# S3 配置
S3_CONFIG = {
    "bucket": os.getenv("S3_BUCKET", "spark-crawler"),
    "region": os.getenv("AWS_REGION", "us-east-1"),
    "prefix": "raw-pages/"
}

# Kafka 配置
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_SERVERS", "localhost:9092"),
    "topics": {
        "urls": "spark.urls",
        "pages": "spark.pages",
        "items": "spark.items"
    }
}

# Prometheus 监控配置
PROMETHEUS_CONFIG = {
    "port": int(os.getenv("PROMETHEUS_PORT", 8000)),
    "path": "/metrics"
}

# 爬虫配置
CRAWLER_CONFIG = {
    "concurrency": int(os.getenv("CRAWLER_CONCURRENCY", 10)),
    "timeout": int(os.getenv("CRAWLER_TIMEOUT", 30)),
    "retry_times": int(os.getenv("CRAWLER_RETRY_TIMES", 3)),
    "download_path": os.getenv("DOWNLOAD_PATH", "./downloads"),
}

# 代理池配置
PROXY_CONFIG = {
    "min_proxies": 20,
    "max_proxies": 100,
    "proxy_sources": [
        "https://proxylist.geonode.com/api/proxy-list",
        "http://pubproxy.com/api/proxy"
    ],
    "check_interval": 300,  # 5分钟检查一次
    "proxy_score": {
        "min": 0.0,
        "max": 10.0,
        "initial": 5.0
    }
}

# Bloom Filter 配置
BLOOM_CONFIG = {
    "capacity": 100_000_000,  # 预期URL数量
    "error_rate": 0.001      # 错误率
}

# 请求限速配置
RATE_LIMIT = {
    'default': 1,  # 每秒请求数
    'example.com': 0.5  # 特定域名限速
}

# 爬虫规则配置
CRAWL_RULES: Dict[str, Any] = {
    "default": {
        "allowed_domains": [],
        "start_urls": [],
        "rules": [
            {
                "allow": r".*",
                "deny": r".*\.(jpg|png|gif|pdf|zip)$",
                "follow": True
            }
        ],
        "custom_settings": {}
    }
}

# Elasticsearch配置
ES_CONFIG = {
    "hosts": ["http://localhost:9200"],
    "index_prefix": "spark_crawler",
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "refresh_interval": "30s"
    }
}

# PageRank配置
PAGERANK_CONFIG = {
    "damping_factor": 0.85,  # 阻尼系数
    "min_iterations": 30,    # 最小迭代次数
    "max_iterations": 100,   # 最大迭代次数
    "convergence_threshold": 1e-6,  # 收敛阈值
    "recalculate_interval": 100,    # 每爬取多少页面重新计算一次
}

# 现代搜索引擎排名配置
RANKING_CONFIG = {
    # PageRank相关配置
    "pagerank": {
        "damping_factor": 0.85,
        "convergence_threshold": 1e-6,
        "max_iterations": 100
    },
    
    # 各维度权重配置
    "ranking_weights": {
        "pagerank": 0.3,    # 基础PageRank权重
        "content": 0.2,     # 内容质量权重
        "time": 0.1,        # 时效性权重
        "user": 0.15,       # 用户行为权重
        "technical": 0.1,   # 技术实现权重
        "authority": 0.15   # 域名权威度权重
    },
    
    # 内容质量评分配置
    "content_quality": {
        "min_length": 500,      # 最小内容长度
        "optimal_length": 2000,  # 最佳内容长度
        "image_ratio": 0.2,     # 图片比例系数
        "link_ratio": 0.1       # 链接比例系数
    },
    
    # 时效性配置
    "freshness": {
        "recent_threshold": 7,    # 最近（天）
        "fresh_threshold": 30,    # 新鲜（天）
        "normal_threshold": 90,   # 普通（天）
        "old_threshold": 365      # 较旧（天）
    },
    
    # 用户行为配置
    "user_metrics": {
        "bounce_rate_threshold": 0.8,  # 跳出率阈值
        "avg_time_cap": 300,          # 平均访问时长上限（秒）
        "ctr_multiplier": 10          # 点击率系数
    },
    
    # 技术评分配置
    "technical": {
        "load_time_thresholds": {
            "excellent": 1000,  # 1秒内
            "good": 2000,      # 2秒内
            "fair": 3000       # 3秒内
        }
    },
    
    # 域名权威度配置
    "authority": {
        "age_cap": 10,              # 域名年龄上限（年）
        "backlinks_cap": 1000,      # 反向链接上限
        "social_signals_cap": 10000  # 社交信号上限
    }
}

# 更新Item Pipeline配置
ITEM_PIPELINES = {
    'pipeline.pipelines.ValidationPipeline': 100,
    'pipeline.pipelines.DuplicatesPipeline': 200,
    'pipeline.pipelines.MongoDBPipeline': 300,
    'pipeline.pipelines.S3Pipeline': 400,
    'pipeline.pipelines.ImagesPipeline': 500,
    'pipeline.index_pipeline.IndexPipeline': 600,
    'pipeline.ranking_pipeline.ModernRankingPipeline': 700  # 使用新的排名管道
} 