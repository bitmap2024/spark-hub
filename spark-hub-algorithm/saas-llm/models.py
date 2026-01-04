import os
import time
import random
import logging
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Generator, Callable
import requests
from openai import OpenAI
import anthropic
import google.generativeai as genai
import zhipuai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_system")

class LLMProvider(ABC):
    """LLM提供商基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model")
        # 支持环境变量替换
        api_key_config = config.get("api_key", "")
        if api_key_config.startswith("${") and api_key_config.endswith("}"):
            env_var = api_key_config[2:-1]
            self.api_key = os.environ.get(env_var)
        else:
            self.api_key = api_key_config
        self.base_url = config.get("base_url")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 1.0)
        self.timeout = config.get("timeout", 30)
        self.priority = config.get("priority", 999)
        self.last_used = 0
        self.failures = 0
        self.is_healthy = True
        self.cooldown_until = 0
        
    @abstractmethod
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """生成文本"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """聊天"""
        pass
    
    def mark_failure(self):
        """标记失败"""
        self.failures += 1
        if self.failures >= 3:
            self.is_healthy = False
            self.cooldown_until = time.time() + 60  # 冷却1分钟
    
    def mark_success(self):
        """标记成功"""
        self.failures = 0
        self.is_healthy = True
        self.last_used = time.time()
    
    def is_available(self) -> bool:
        """检查是否可用"""
        if not self.is_healthy:
            if time.time() < self.cooldown_until:
                return False
            self.is_healthy = True  # 冷却期结束，恢复健康状态
        return True
    
    def get_priority(self) -> int:
        """获取优先级"""
        return self.priority


class OpenAIProvider(LLMProvider):
    """OpenAI提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.frequency_penalty),
                presence_penalty=kwargs.get("presence_penalty", self.presence_penalty),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                self.mark_success()
                return stream_generator()
            else:
                result = response.choices[0].message.content
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            self.mark_failure()
            raise
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.frequency_penalty),
                presence_penalty=kwargs.get("presence_penalty", self.presence_penalty),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                self.mark_success()
                return stream_generator()
            else:
                result = response.choices[0].message.content
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            self.mark_failure()
            raise


class QwenProvider(LLMProvider):
    """阿里云通义千问提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.enable_search = config.get("enable_search", False)
        self.enable_thinking = config.get("enable_thinking", False)
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            extra_body = {}
            if self.enable_search or kwargs.get("enable_search", False):
                extra_body["enable_search"] = True
            if self.enable_thinking or kwargs.get("enable_thinking", False):
                extra_body["enable_thinking"] = True
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.frequency_penalty),
                presence_penalty=kwargs.get("presence_penalty", self.presence_penalty),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout),
                extra_body=extra_body if extra_body else None
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                        # 支持思考过程输出
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            yield f"[思考过程] {chunk.choices[0].delta.reasoning_content}"
                self.mark_success()
                return stream_generator()
            else:
                result = response.choices[0].message.content
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Qwen API error: {str(e)}")
            self.mark_failure()
            raise
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            extra_body = {}
            if self.enable_search or kwargs.get("enable_search", False):
                extra_body["enable_search"] = True
            if self.enable_thinking or kwargs.get("enable_thinking", False):
                extra_body["enable_thinking"] = True
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.frequency_penalty),
                presence_penalty=kwargs.get("presence_penalty", self.presence_penalty),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout),
                extra_body=extra_body if extra_body else None
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                        # 支持思考过程输出
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            yield f"[思考过程] {chunk.choices[0].delta.reasoning_content}"
                self.mark_success()
                return stream_generator()
            else:
                result = response.choices[0].message.content
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Qwen API error: {str(e)}")
            self.mark_failure()
            raise


class SiliconFlowProvider(LLMProvider):
    """硅基流动提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.top_k = config.get("top_k", 50)
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.frequency_penalty),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout),
                extra_body={
                    "top_k": kwargs.get("top_k", self.top_k)
                }
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                        # 支持推理模型的思考过程
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            yield f"[推理过程] {chunk.choices[0].delta.reasoning_content}"
                self.mark_success()
                return stream_generator()
            else:
                result = response.choices[0].message.content
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"SiliconFlow API error: {str(e)}")
            self.mark_failure()
            raise
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.frequency_penalty),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout),
                extra_body={
                    "top_k": kwargs.get("top_k", self.top_k)
                }
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                        # 支持推理模型的思考过程
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            yield f"[推理过程] {chunk.choices[0].delta.reasoning_content}"
                self.mark_success()
                return stream_generator()
            else:
                result = response.choices[0].message.content
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"SiliconFlow API error: {str(e)}")
            self.mark_failure()
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.delta.text:
                            yield chunk.delta.text
                self.mark_success()
                return stream_generator()
            else:
                result = response.content[0].text
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            self.mark_failure()
            raise
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            # 转换消息格式
            anthropic_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "assistant"
                anthropic_messages.append({"role": role, "content": msg["content"]})
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                messages=anthropic_messages,
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.delta.text:
                            yield chunk.delta.text
                self.mark_success()
                return stream_generator()
            else:
                result = response.content[0].text
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            self.mark_failure()
            raise


class GoogleProvider(LLMProvider):
    """Google提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(config.get("model"))
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                ),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                self.mark_success()
                return stream_generator()
            else:
                result = response.text
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            self.mark_failure()
            raise
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            # 转换消息格式
            chat = self.model.start_chat()
            for msg in messages[:-1]:  # 除了最后一条消息
                if msg["role"] == "user":
                    chat.send_message(msg["content"])
                else:
                    # 模拟助手回复
                    chat.history.append({"role": "assistant", "parts": [msg["content"]]})
            
            # 发送最后一条消息并获取回复
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                ),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                self.mark_success()
                return stream_generator()
            else:
                result = response.text
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            self.mark_failure()
            raise


class ZhipuProvider(LLMProvider):
    """智谱AI提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        zhipuai.api_key = self.api_key
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            response = zhipuai.model_api.invoke(
                model=self.model,
                prompt=prompt,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.get("data", {}).get("text"):
                            yield chunk["data"]["text"]
                self.mark_success()
                return stream_generator()
            else:
                result = response["data"]["text"]
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Zhipu API error: {str(e)}")
            self.mark_failure()
            raise
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        try:
            # 转换消息格式
            zhipu_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "assistant"
                zhipu_messages.append({"role": role, "content": msg["content"]})
            
            response = zhipuai.model_api.invoke(
                model=self.model,
                messages=zhipu_messages,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=stream,
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.get("data", {}).get("text"):
                            yield chunk["data"]["text"]
                self.mark_success()
                return stream_generator()
            else:
                result = response["data"]["text"]
                self.mark_success()
                return result
        except Exception as e:
            logger.error(f"Zhipu API error: {str(e)}")
            self.mark_failure()
            raise


class LLMProxyPool:
    """LLM代理池"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.providers = self._initialize_providers()
        self.strategy = self.config["proxy_pool"]["strategy"]
        self.failover = self.config["proxy_pool"]["failover"]
        self.max_retries = self.config["proxy_pool"]["max_retries"]
        self.cooldown_period = self.config["proxy_pool"]["cooldown_period"]
        self.health_check_interval = self.config["proxy_pool"]["health_check_interval"]
        self.last_health_check = 0
        self.round_robin_index = 0
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
    
    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """初始化提供商"""
        providers = {}
        provider_map = {
            "openai": OpenAIProvider,
            "qwen": QwenProvider,
            "siliconflow": SiliconFlowProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider
        }
        
        for model_name, model_config in self.config["models"].items():
            provider_name = model_config["provider"]
            if provider_name in provider_map:
                try:
                    providers[model_name] = provider_map[provider_name](model_config)
                    logger.info(f"Initialized provider {provider_name} for model {model_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize provider {provider_name} for model {model_name}: {str(e)}")
        
        return providers
    
    def _select_provider(self) -> Optional[LLMProvider]:
        """选择提供商"""
        available_providers = [p for p in self.providers.values() if p.is_available()]
        
        if not available_providers:
            logger.warning("No available providers")
            return None
        
        if self.strategy == "round_robin":
            provider = available_providers[self.round_robin_index % len(available_providers)]
            self.round_robin_index += 1
            return provider
        elif self.strategy == "random":
            return random.choice(available_providers)
        elif self.strategy == "priority":
            return min(available_providers, key=lambda p: p.get_priority())
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, falling back to priority")
            return min(available_providers, key=lambda p: p.get_priority())
    
    def _health_check(self):
        """健康检查"""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        logger.info("Running health check on providers")
        
        for name, provider in self.providers.items():
            if not provider.is_healthy and time.time() >= provider.cooldown_until:
                provider.is_healthy = True
                logger.info(f"Provider {name} is back online")
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """生成文本"""
        self._health_check()
        
        if self.failover:
            for _ in range(self.max_retries):
                provider = self._select_provider()
                if not provider:
                    raise Exception("No available providers")
                
                try:
                    return provider.generate(prompt, stream, **kwargs)
                except Exception as e:
                    logger.warning(f"Provider {provider.model} failed: {str(e)}, trying next provider")
                    continue
            
            raise Exception("All providers failed")
        else:
            provider = self._select_provider()
            if not provider:
                raise Exception("No available providers")
            
            return provider.generate(prompt, stream, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """聊天"""
        self._health_check()
        
        if self.failover:
            for _ in range(self.max_retries):
                provider = self._select_provider()
                if not provider:
                    raise Exception("No available providers")
                
                try:
                    return provider.chat(messages, stream, **kwargs)
                except Exception as e:
                    logger.warning(f"Provider {provider.model} failed: {str(e)}, trying next provider")
                    continue
            
            raise Exception("All providers failed")
        else:
            provider = self._select_provider()
            if not provider:
                raise Exception("No available providers")
            
            return provider.chat(messages, stream, **kwargs) 