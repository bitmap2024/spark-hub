import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Union, Generator
from models import LLMProxyPool

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_system")

class LLMClient:
    """LLM客户端"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.proxy_pool = LLMProxyPool(config_path)
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """生成文本"""
        return self.proxy_pool.generate(prompt, stream, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """聊天"""
        return self.proxy_pool.chat(messages, stream, **kwargs)


def print_stream(stream: Generator[str, None, None]):
    """打印流式输出"""
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM系统")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["generate", "chat"], default="chat", help="模式：generate或chat")
    parser.add_argument("--stream", action="store_true", help="是否使用流式输出")
    parser.add_argument("--prompt", type=str, help="生成模式的提示词")
    parser.add_argument("--messages", type=str, help="聊天模式的消息，JSON格式")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--max_tokens", type=int, default=4096, help="最大token数")
    
    args = parser.parse_args()
    
    client = LLMClient(args.config)
    
    if args.mode == "generate":
        if not args.prompt:
            print("生成模式需要提供--prompt参数")
            return
        
        if args.stream:
            stream = client.generate(args.prompt, stream=True, temperature=args.temperature, max_tokens=args.max_tokens)
            print_stream(stream)
        else:
            result = client.generate(args.prompt, stream=False, temperature=args.temperature, max_tokens=args.max_tokens)
            print(result)
    
    elif args.mode == "chat":
        if not args.messages:
            # 交互式聊天
            messages = []
            print("开始聊天（输入'exit'退出）：")
            
            while True:
                user_input = input("\n用户: ")
                if user_input.lower() == "exit":
                    break
                
                messages.append({"role": "user", "content": user_input})
                
                if args.stream:
                    stream = client.chat(messages, stream=True, temperature=args.temperature, max_tokens=args.max_tokens)
                    print("\n助手: ", end="")
                    print_stream(stream)
                else:
                    result = client.chat(messages, stream=False, temperature=args.temperature, max_tokens=args.max_tokens)
                    print("\n助手:", result)
                
                messages.append({"role": "assistant", "content": result if not args.stream else "".join(list(stream))})
        else:
            # 从JSON文件加载消息
            try:
                with open(args.messages, "r", encoding="utf-8") as f:
                    messages = json.load(f)
                
                if args.stream:
                    stream = client.chat(messages, stream=True, temperature=args.temperature, max_tokens=args.max_tokens)
                    print_stream(stream)
                else:
                    result = client.chat(messages, stream=False, temperature=args.temperature, max_tokens=args.max_tokens)
                    print(result)
            except Exception as e:
                print(f"加载消息失败: {str(e)}")


if __name__ == "__main__":
    main()
