"""
反爬虫机制模块
"""
import random
import time
from typing import Dict, List, Optional, Union
from fake_useragent import UserAgent
from urllib.parse import urlparse
import hashlib
import json
from datetime import datetime, timedelta
import aiohttp
import asyncio
from collections import defaultdict
from user_agents import parse
import logging
from .optimization import ResourcePool
import math

class UserAgentManager:
    """User-Agent管理器"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.browser_types = ['chrome', 'firefox', 'safari', 'opera']
        self.custom_uas = []
        
    def get_random_ua(self) -> str:
        """获取随机UA"""
        if random.random() < 0.8:  # 80%概率使用预定义UA
            browser = random.choice(self.browser_types)
            return getattr(self.ua, browser)
        return random.choice(self.custom_uas) if self.custom_uas else self.ua.random
        
    def add_custom_ua(self, ua: str):
        """添加自定义UA"""
        self.custom_uas.append(ua)

class CookieManager:
    """Cookie管理器"""
    
    def __init__(self):
        self.cookies: Dict[str, List[Dict]] = defaultdict(list)
        self.max_cookies_per_domain = 10
        
    def add_cookie(self, domain: str, cookie: Dict):
        """添加Cookie"""
        if len(self.cookies[domain]) >= self.max_cookies_per_domain:
            self.cookies[domain].pop(0)
        self.cookies[domain].append(cookie)
        
    def get_cookie(self, domain: str) -> Optional[Dict]:
        """获取Cookie"""
        domain_cookies = self.cookies.get(domain, [])
        return random.choice(domain_cookies) if domain_cookies else None
        
    def rotate_cookies(self, domain: str):
        """轮换Cookie"""
        if domain in self.cookies and len(self.cookies[domain]) > 1:
            self.cookies[domain] = self.cookies[domain][1:] + [self.cookies[domain][0]]

class ProxyRotator:
    """代理轮换器"""
    
    def __init__(self, proxy_pool: ResourcePool):
        self.proxy_pool = proxy_pool
        self.proxy_scores = defaultdict(lambda: 5.0)  # 默认分数5.0
        self.max_score = 10.0
        self.min_score = 0.0
        self.success_rate = defaultdict(lambda: {'success': 0, 'total': 0})
        self.last_used = defaultdict(float)
        self.min_success_rate = 0.3  # 最低成功率阈值
        
    async def get_proxy(self) -> Optional[str]:
        """获取代理"""
        now = time.time()
        # 清理低分和低成功率的代理
        await self._clean_poor_proxies()
        
        # 获取可用代理
        proxy = await self.proxy_pool.get()
        if proxy:
            self.last_used[proxy] = now
        return proxy
        
    def update_proxy_score(self, proxy: str, success: bool):
        """更新代理分数"""
        if success:
            self.proxy_scores[proxy] = min(
                self.proxy_scores[proxy] + 0.5,
                self.max_score
            )
            self.success_rate[proxy]['success'] += 1
        else:
            self.proxy_scores[proxy] = max(
                self.proxy_scores[proxy] - 1.0,
                self.min_score
            )
        self.success_rate[proxy]['total'] += 1
        
    async def _clean_poor_proxies(self):
        """清理表现不佳的代理"""
        now = time.time()
        expired_time = 3600  # 1小时未使用就清理
        
        for proxy in list(self.proxy_scores.keys()):
            rate = self.success_rate[proxy]
            success_rate = rate['success'] / rate['total'] if rate['total'] > 0 else 0
            
            # 清理低分、低成功率或长期未使用的代理
            if (self.proxy_scores[proxy] < 2.0 or 
                success_rate < self.min_success_rate or
                now - self.last_used[proxy] > expired_time):
                
                del self.proxy_scores[proxy]
                del self.success_rate[proxy]
                del self.last_used[proxy]
                await self.proxy_pool.remove(proxy)
                
    async def get_proxy_stats(self) -> Dict:
        """获取代理统计信息"""
        stats = {
            'total_proxies': len(self.proxy_scores),
            'avg_score': sum(self.proxy_scores.values()) / len(self.proxy_scores) if self.proxy_scores else 0,
            'success_rates': {
                proxy: rate['success'] / rate['total'] if rate['total'] > 0 else 0
                for proxy, rate in self.success_rate.items()
            }
        }
        return stats

class RequestThrottler:
    """请求限速器"""
    
    def __init__(self):
        self.domain_requests = defaultdict(list)
        self.max_requests_per_domain = defaultdict(lambda: 10)
        self.time_window = 60  # 60秒时间窗口
        self.domain_stats = defaultdict(lambda: {
            'success_count': 0,
            'fail_count': 0,
            'avg_response_time': 0
        })
        self.request_queue = defaultdict(asyncio.Queue)
        
    async def should_throttle(self, domain: str) -> bool:
        """判断是否需要限速"""
        now = time.time()
        # 清理过期请求记录
        self.domain_requests[domain] = [
            t for t in self.domain_requests[domain]
            if now - t < self.time_window
        ]
        
        # 根据域名统计信息动态调整限速
        if self.domain_stats[domain]['fail_count'] > 0:
            fail_rate = self.domain_stats[domain]['fail_count'] / (
                self.domain_stats[domain]['success_count'] + 
                self.domain_stats[domain]['fail_count']
            )
            if fail_rate > 0.2:  # 失败率超过20%
                self.max_requests_per_domain[domain] = max(
                    5,  # 最小请求数
                    self.max_requests_per_domain[domain] // 2  # 减半
                )
        
        return len(self.domain_requests[domain]) >= self.max_requests_per_domain[domain]
        
    async def add_request(self, domain: str, response_time: float = None, success: bool = True):
        """添加请求记录"""
        now = time.time()
        self.domain_requests[domain].append(now)
        
        # 更新统计信息
        if success:
            self.domain_stats[domain]['success_count'] += 1
        else:
            self.domain_stats[domain]['fail_count'] += 1
            
        if response_time is not None:
            old_avg = self.domain_stats[domain]['avg_response_time']
            total_requests = (self.domain_stats[domain]['success_count'] + 
                            self.domain_stats[domain]['fail_count'])
            self.domain_stats[domain]['avg_response_time'] = (
                (old_avg * (total_requests - 1) + response_time) / total_requests
            )
            
    def set_rate_limit(self, domain: str, max_requests: int):
        """设置域名请求限制"""
        self.max_requests_per_domain[domain] = max_requests
        
    async def wait_for_slot(self, domain: str) -> None:
        """等待请求槽位"""
        while await self.should_throttle(domain):
            await asyncio.sleep(0.1)
            
    def get_domain_stats(self, domain: str) -> Dict:
        """获取域名统计信息"""
        return {
            'current_requests': len(self.domain_requests[domain]),
            'max_requests': self.max_requests_per_domain[domain],
            'success_count': self.domain_stats[domain]['success_count'],
            'fail_count': self.domain_stats[domain]['fail_count'],
            'avg_response_time': self.domain_stats[domain]['avg_response_time']
        }

class BehaviorSimulator:
    """用户行为模拟器"""
    
    def __init__(self):
        self.min_delay = 1
        self.max_delay = 5
        self.scroll_probability = 0.7
        self.click_probability = 0.3
        self.mouse_move_probability = 0.5
        self.keyboard_input_probability = 0.2
        self.viewport_height = 800
        self.typing_speed = (50, 200)  # 每分钟字符数范围
        
    async def simulate_human_behavior(self, page) -> None:
        """模拟人类行为"""
        # 随机延迟
        await self._random_delay()
        
        # 模拟鼠标移动
        if random.random() < self.mouse_move_probability:
            await self._simulate_mouse_movement(page)
        
        # 模拟滚动
        if random.random() < self.scroll_probability:
            await self._simulate_scroll(page)
        
        # 模拟随机点击
        if random.random() < self.click_probability:
            await self._simulate_click(page)
            
        # 模拟键盘输入
        if random.random() < self.keyboard_input_probability:
            await self._simulate_keyboard_input(page)
            
    async def _random_delay(self):
        """随机延迟"""
        delay = random.uniform(self.min_delay, self.max_delay)
        # 添加微小的随机变化
        delay += random.gauss(0, 0.1)
        await asyncio.sleep(max(0, delay))
        
    async def _simulate_mouse_movement(self, page):
        """模拟鼠标移动"""
        # 获取页面尺寸
        page_dimensions = await page.evaluate('''
            () => ({
                width: document.documentElement.clientWidth,
                height: document.documentElement.clientHeight
            })
        ''')
        
        # 生成随机的鼠标移动路径
        points = self._generate_bezier_curve(
            (0, 0),
            (random.randint(0, page_dimensions['width']),
             random.randint(0, page_dimensions['height']))
        )
        
        # 执行鼠标移动
        for point in points:
            await page.mouse.move(point[0], point[1])
            await asyncio.sleep(random.uniform(0.01, 0.03))
            
    async def _simulate_scroll(self, page):
        """模拟滚动行为"""
        # 获取页面高度
        total_height = await page.evaluate('document.body.scrollHeight')
        current_position = await page.evaluate('window.pageYOffset')
        
        # 计算滚动目标
        max_scroll = min(
            current_position + random.randint(300, 1000),
            total_height - self.viewport_height
        )
        
        # 平滑滚动
        steps = random.randint(5, 15)
        for i in range(steps):
            next_pos = current_position + (max_scroll - current_position) * (i + 1) / steps
            await page.evaluate(f'window.scrollTo(0, {next_pos})')
            await asyncio.sleep(random.uniform(0.05, 0.15))
            
    async def _simulate_click(self, page):
        """模拟点击行为"""
        elements = await page.query_selector_all('a, button, input[type="submit"]')
        if elements:
            element = random.choice(elements)
            # 先hover
            await element.hover()
            await asyncio.sleep(random.uniform(0.2, 0.8))
            # 模拟人类点击延迟
            await asyncio.sleep(random.uniform(0.1, 0.3))
            await element.click()
            
    async def _simulate_keyboard_input(self, page):
        """模拟键盘输入"""
        input_elements = await page.query_selector_all('input[type="text"], textarea')
        if input_elements:
            input_element = random.choice(input_elements)
            await input_element.click()
            
            # 生成随机文本
            text = self._generate_random_text()
            
            # 模拟人类打字速度
            for char in text:
                await input_element.type(char)
                # 使用正态分布模拟打字间隔
                delay = random.gauss(
                    60 / random.uniform(*self.typing_speed),  # 基础延迟
                    0.02  # 标准差
                )
                await asyncio.sleep(max(0.01, delay))
                
    def _generate_random_text(self) -> str:
        """生成随机文本"""
        words = ['hello', 'world', 'test', 'input', 'search', 'query']
        return ' '.join(random.choices(words, k=random.randint(1, 4)))
        
    def _generate_bezier_curve(self, start: tuple, end: tuple, control_points: int = 2) -> List[tuple]:
        """生成贝塞尔曲线路径"""
        points = [start]
        # 生成控制点
        for _ in range(control_points):
            points.append((
                random.randint(min(start[0], end[0]), max(start[0], end[0])),
                random.randint(min(start[1], end[1]), max(start[1], end[1]))
            ))
        points.append(end)
        
        # 生成曲线上的点
        curve_points = []
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            x = y = 0
            n = len(points) - 1
            for j in range(n + 1):
                factor = (
                    math.factorial(n) /
                    (math.factorial(j) * math.factorial(n - j))
                ) * (t ** j) * ((1 - t) ** (n - j))
                x += points[j][0] * factor
                y += points[j][1] * factor
            curve_points.append((int(x), int(y)))
        
        return curve_points

class FingerprintManager:
    """浏览器指纹管理器"""
    
    def __init__(self):
        self.fingerprints = []
        self.current_index = 0
        
    def add_fingerprint(self, fingerprint: Dict):
        """添加浏览器指纹"""
        self.fingerprints.append(fingerprint)
        
    def get_next_fingerprint(self) -> Optional[Dict]:
        """获取下一个指纹"""
        if not self.fingerprints:
            return None
        fingerprint = self.fingerprints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.fingerprints)
        return fingerprint

class AntiCrawlerManager:
    """反爬虫管理器"""
    
    def __init__(self, proxy_pool: ResourcePool):
        self.ua_manager = UserAgentManager()
        self.cookie_manager = CookieManager()
        self.proxy_rotator = ProxyRotator(proxy_pool)
        self.throttler = RequestThrottler()
        self.behavior_simulator = BehaviorSimulator()
        self.fingerprint_manager = FingerprintManager()
        
        # 请求头模板
        self.headers_template = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # 监控指标
        self.metrics = defaultdict(lambda: {
            'total_requests': 0,
            'success_requests': 0,
            'failed_requests': 0,
            'blocked_requests': 0,
            'avg_response_time': 0,
            'start_time': time.time()
        })
        
        # 配置
        self.config = {
            'max_retries': 3,
            'retry_delay': 5,
            'success_codes': {200, 201, 202, 203, 204, 206, 207, 208, 226},
            'block_codes': {403, 429, 430},
            'suspicious_patterns': [
                'robot', 'bot', 'crawler', 'spider',
                'banned', 'blocked', 'captcha'
            ]
        }
        
    def get_headers(self, url: str) -> Dict:
        """获取请求头"""
        domain = urlparse(url).netloc
        headers = self.headers_template.copy()
        headers['User-Agent'] = self.ua_manager.get_random_ua()
        
        # 添加Cookie
        cookie = self.cookie_manager.get_cookie(domain)
        if cookie:
            headers['Cookie'] = self._dict_to_cookie_str(cookie)
            
        # 添加随机的请求头
        if random.random() < 0.3:  # 30%概率添加额外头
            headers['Sec-Fetch-Dest'] = random.choice(['document', 'image', 'script'])
            headers['Sec-Fetch-Mode'] = random.choice(['navigate', 'no-cors'])
            headers['Sec-Fetch-Site'] = random.choice(['same-origin', 'cross-site'])
            
        # 添加浏览器指纹
        fingerprint = self.fingerprint_manager.get_next_fingerprint()
        if fingerprint:
            headers.update(fingerprint)
            
        return headers
        
    def _dict_to_cookie_str(self, cookie_dict: Dict) -> str:
        """将Cookie字典转换为字符串"""
        return '; '.join(f'{k}={v}' for k, v in cookie_dict.items())
        
    async def prepare_request(self, url: str) -> Dict:
        """准备请求参数"""
        domain = urlparse(url).netloc
        
        # 检查是否需要限速
        await self.throttler.wait_for_slot(domain)
            
        # 获取代理
        proxy = await self.proxy_rotator.get_proxy()
        
        # 准备请求参数
        request_params = {
            'headers': self.get_headers(url),
            'proxy': proxy if proxy else None,
            'timeout': aiohttp.ClientTimeout(total=30)
        }
        
        # 记录请求
        await self.throttler.add_request(domain)
        self.metrics[domain]['total_requests'] += 1
        
        return request_params
        
    async def handle_response(
        self,
        url: str,
        response: aiohttp.ClientResponse,
        proxy: Optional[str],
        response_time: float
    ):
        """处理响应"""
        domain = urlparse(url).netloc
        
        # 更新代理分数
        if proxy:
            success = response.status in self.config['success_codes']
            self.proxy_rotator.update_proxy_score(proxy, success)
            
        # 提取并保存Cookie
        if response.cookies:
            self.cookie_manager.add_cookie(
                domain,
                dict(response.cookies)
            )
            
        # 更新统计信息
        if response.status in self.config['success_codes']:
            self.metrics[domain]['success_requests'] += 1
        elif response.status in self.config['block_codes']:
            self.metrics[domain]['blocked_requests'] += 1
        else:
            self.metrics[domain]['failed_requests'] += 1
            
        # 更新平均响应时间
        old_avg = self.metrics[domain]['avg_response_time']
        total_requests = self.metrics[domain]['total_requests']
        self.metrics[domain]['avg_response_time'] = (
            (old_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # 检查是否被封禁
        if response.status in self.config['block_codes']:
            await self._handle_blocking(domain)
            
        # 轮换Cookie
        self.cookie_manager.rotate_cookies(domain)
        
    async def _handle_blocking(self, domain: str):
        """处理被封禁情况"""
        # 增加限速
        current_limit = self.throttler.max_requests_per_domain[domain]
        self.throttler.set_rate_limit(domain, max(1, current_limit // 2))
        
        # 清理该域名的Cookie
        self.cookie_manager.cookies[domain] = []
        
        # 等待一段时间
        await asyncio.sleep(self.config['retry_delay'])
        
    def get_metrics(self, domain: Optional[str] = None) -> Dict:
        """获取监控指标"""
        if domain:
            metrics = self.metrics[domain].copy()
            metrics['uptime'] = time.time() - metrics['start_time']
            return metrics
            
        # 返回所有域名的汇总指标
        total_metrics = {
            'total_requests': 0,
            'success_requests': 0,
            'failed_requests': 0,
            'blocked_requests': 0,
            'domains_count': len(self.metrics),
            'start_time': min(m['start_time'] for m in self.metrics.values())
        }
        
        for domain_metrics in self.metrics.values():
            total_metrics['total_requests'] += domain_metrics['total_requests']
            total_metrics['success_requests'] += domain_metrics['success_requests']
            total_metrics['failed_requests'] += domain_metrics['failed_requests']
            total_metrics['blocked_requests'] += domain_metrics['blocked_requests']
            
        total_metrics['uptime'] = time.time() - total_metrics['start_time']
        total_metrics['success_rate'] = (
            total_metrics['success_requests'] / total_metrics['total_requests']
            if total_metrics['total_requests'] > 0 else 0
        )
        
        return total_metrics
        
    async def simulate_browser(self, page) -> None:
        """模拟浏览器行为"""
        await self.behavior_simulator.simulate_human_behavior(page)
        
    def update_fingerprint(self) -> None:
        """更新浏览器指纹"""
        fingerprint = self.fingerprint_manager.get_next_fingerprint()
        if fingerprint:
            # 更新浏览器指纹相关的设置
            pass 