from typing import Dict, Any, List, Set
import numpy as np
from collections import defaultdict
from urllib.parse import urlparse
from datetime import datetime
from loguru import logger
import networkx as nx
from redis import Redis
from config.settings import REDIS_CONFIG, PAGERANK_CONFIG

class PageRankPipeline:
    """PageRank计算管道"""
    
    def __init__(
        self,
        redis_url: str = REDIS_CONFIG["url"],
        damping_factor: float = PAGERANK_CONFIG["damping_factor"],
        min_iterations: int = PAGERANK_CONFIG["min_iterations"],
        max_iterations: int = PAGERANK_CONFIG["max_iterations"],
        convergence_threshold: float = PAGERANK_CONFIG["convergence_threshold"]
    ):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.damping_factor = damping_factor
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.graph = nx.DiGraph()  # 有向图
        
    def get_domain(self, url: str) -> str:
        """获取URL的域名"""
        return urlparse(url).netloc
        
    def process_item(self, item: Dict[str, Any], spider) -> Dict[str, Any]:
        """处理爬取项，更新链接关系"""
        try:
            current_url = item["url"]
            outgoing_links = item.get("links", [])
            
            # 更新图结构
            if not self.graph.has_node(current_url):
                self.graph.add_node(
                    current_url,
                    domain=self.get_domain(current_url),
                    title=item.get("title", ""),
                    last_updated=datetime.now().isoformat()
                )
            
            # 添加出链
            for link in outgoing_links:
                if not self.graph.has_node(link):
                    self.graph.add_node(
                        link,
                        domain=self.get_domain(link),
                        title="",  # 暂未爬取
                        last_updated=datetime.now().isoformat()
                    )
                self.graph.add_edge(current_url, link)
            
            # 定期计算PageRank
            if self.should_calculate_pagerank():
                self.calculate_pagerank()
            
            # 将当前页面的PageRank值添加到item中
            item["pagerank"] = self.get_pagerank(current_url)
            return item
            
        except Exception as e:
            logger.error(f"Error processing PageRank: {str(e)}")
            return item
    
    def should_calculate_pagerank(self) -> bool:
        """判断是否需要重新计算PageRank"""
        # 每100个新节点计算一次
        return self.graph.number_of_nodes() % 100 == 0
    
    def calculate_pagerank(self):
        """计算整个图的PageRank值"""
        try:
            # 使用NetworkX内置的PageRank算法
            pagerank = nx.pagerank(
                self.graph,
                alpha=self.damping_factor,
                tol=self.convergence_threshold,
                max_iter=self.max_iterations
            )
            
            # 将结果存入Redis
            pipeline = self.redis.pipeline()
            for url, rank in pagerank.items():
                pipeline.hset(
                    "pagerank:scores",
                    url,
                    float(rank)
                )
            pipeline.execute()
            
            # 计算并存储域名级别的PageRank
            domain_ranks = defaultdict(float)
            for url, rank in pagerank.items():
                domain = self.get_domain(url)
                domain_ranks[domain] += rank
            
            # 存储域名级别的PageRank
            pipeline = self.redis.pipeline()
            for domain, rank in domain_ranks.items():
                pipeline.hset(
                    "pagerank:domain_scores",
                    domain,
                    float(rank)
                )
            pipeline.execute()
            
            logger.info(
                f"Calculated PageRank for {len(pagerank)} pages "
                f"and {len(domain_ranks)} domains"
            )
            
        except Exception as e:
            logger.error(f"Error calculating PageRank: {str(e)}")
    
    def get_pagerank(self, url: str) -> float:
        """获取指定URL的PageRank值"""
        try:
            rank = self.redis.hget("pagerank:scores", url)
            return float(rank) if rank else 0.0
        except:
            return 0.0
    
    def get_domain_rank(self, domain: str) -> float:
        """获取指定域名的PageRank值"""
        try:
            rank = self.redis.hget("pagerank:domain_scores", domain)
            return float(rank) if rank else 0.0
        except:
            return 0.0
    
    def get_top_pages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取PageRank最高的页面"""
        try:
            # 获取所有分数
            all_scores = self.redis.hgetall("pagerank:scores")
            
            # 排序并限制数量
            top_pages = sorted(
                all_scores.items(),
                key=lambda x: float(x[1]),
                reverse=True
            )[:limit]
            
            # 构建结果
            result = []
            for url, rank in top_pages:
                node_data = self.graph.nodes[url]
                result.append({
                    "url": url,
                    "pagerank": float(rank),
                    "title": node_data.get("title", ""),
                    "domain": node_data.get("domain", ""),
                    "last_updated": node_data.get("last_updated", "")
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting top pages: {str(e)}")
            return []
    
    def get_domain_authority(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取PageRank最高的域名"""
        try:
            # 获取所有域名分数
            all_scores = self.redis.hgetall("pagerank:domain_scores")
            
            # 排序并限制数量
            top_domains = sorted(
                all_scores.items(),
                key=lambda x: float(x[1]),
                reverse=True
            )[:limit]
            
            # 构建结果
            return [
                {
                    "domain": domain,
                    "pagerank": float(rank)
                }
                for domain, rank in top_domains
            ]
            
        except Exception as e:
            logger.error(f"Error getting domain authority: {str(e)}")
            return []
    
    def close_spider(self, spider):
        """爬虫关闭时的清理工作"""
        # 最后计算一次PageRank
        self.calculate_pagerank()
        
        # 记录统计信息
        stats = {
            "total_pages": self.graph.number_of_nodes(),
            "total_links": self.graph.number_of_edges(),
            "total_domains": len({
                self.get_domain(url)
                for url in self.graph.nodes
            }),
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis.hmset("pagerank:stats", stats)
        logger.info(f"PageRank pipeline closed: {stats}") 