from typing import Dict, Any, List
from elasticsearch import Elasticsearch
from datetime import datetime
from loguru import logger
from config.settings import ES_CONFIG

class IndexPipeline:
    """倒排索引管道"""
    
    def __init__(
        self,
        es_hosts: List[str] = ES_CONFIG["hosts"],
        index_prefix: str = ES_CONFIG["index_prefix"]
    ):
        self.es = Elasticsearch(es_hosts)
        self.index_prefix = index_prefix
        
    def get_index_name(self, spider_name: str) -> str:
        """获取索引名称"""
        return f"{self.index_prefix}_{spider_name}_{datetime.now().strftime('%Y%m')}"
        
    def create_index_mapping(self, index_name: str):
        """创建索引映射"""
        mapping = {
            "mappings": {
                "properties": {
                    "url": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "tags": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "crawled_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "spider": {"type": "keyword"},
                    "domain": {"type": "keyword"},
                    # 图片相关字段
                    "images": {
                        "type": "nested",
                        "properties": {
                            "url": {"type": "keyword"},
                            "path": {"type": "keyword"},
                            "caption": {"type": "text"}
                        }
                    },
                    # 结构化数据
                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    }
                }
            },
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "refresh_interval": "30s",
                "analysis": {
                    "analyzer": {
                        "html_strip": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"],
                            "char_filter": ["html_strip"]
                        }
                    }
                }
            }
        }
        
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=mapping)
            logger.info(f"Created index: {index_name}")
    
    def process_item(self, item: Dict[str, Any], spider) -> Dict[str, Any]:
        """处理爬取项，建立索引"""
        try:
            index_name = self.get_index_name(spider.name)
            self.create_index_mapping(index_name)
            
            # 准备索引文档
            doc = {
                "url": item["url"],
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "tags": item.get("tags", []),
                "category": item.get("category", ""),
                "crawled_at": item.get("crawled_at", datetime.now().isoformat()),
                "updated_at": datetime.now().isoformat(),
                "spider": spider.name,
                "domain": item.get("domain", ""),
                "images": item.get("images", []),
                "metadata": item.get("metadata", {})
            }
            
            # 索引文档
            self.es.index(
                index=index_name,
                id=item["url"],  # 使用URL作为文档ID
                body=doc,
                refresh=True  # 立即刷新，用于测试环境
            )
            
            logger.info(f"Indexed document: {item['url']}")
            return item
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return item
            
    def close_spider(self, spider):
        """关闭爬虫时的清理工作"""
        if self.es:
            self.es.close() 