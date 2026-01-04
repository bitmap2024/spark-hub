#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM系统使用示例
"""

import os
import json
import time
from main import LLMClient

def test_generate():
    """测试生成功能"""
    print("=" * 50)
    print("测试生成功能")
    print("=" * 50)
    
    client = LLMClient("config.yaml")
    
    # 非流式生成
    print("\n非流式生成:")
    prompt = "请用一句话介绍大语言模型"
    start_time = time.time()
    result = client.generate(prompt)
    end_time = time.time()
    print(f"提示词: {prompt}")
    print(f"结果: {result}")
    print(f"耗时: {end_time - start_time:.2f}秒")
    
    # 流式生成
    print("\n流式生成:")
    prompt = "请用三句话介绍人工智能的发展历史"
    print(f"提示词: {prompt}")
    print("结果: ", end="")
    start_time = time.time()
    stream = client.generate(prompt, stream=True)
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()
    end_time = time.time()
    print(f"耗时: {end_time - start_time:.2f}秒")


def test_chat():
    """测试聊天功能"""
    print("\n" + "=" * 50)
    print("测试聊天功能")
    print("=" * 50)
    
    client = LLMClient("config.yaml")
    
    # 从文件加载消息
    print("\n从文件加载消息:")
    try:
        with open("messages.json", "r", encoding="utf-8") as f:
            messages = json.load(f)
        
        # 非流式聊天
        print("\n非流式聊天:")
        start_time = time.time()
        result = client.chat(messages)
        end_time = time.time()
        print(f"结果: {result}")
        print(f"耗时: {end_time - start_time:.2f}秒")
        
        # 流式聊天
        print("\n流式聊天:")
        print("结果: ", end="")
        start_time = time.time()
        stream = client.chat(messages, stream=True)
        for chunk in stream:
            print(chunk, end="", flush=True)
        print()
        end_time = time.time()
        print(f"耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        print(f"加载消息失败: {str(e)}")
    
    # 交互式聊天
    print("\n交互式聊天:")
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，请用简洁明了的语言回答问题。"}
    ]
    
    print("开始聊天（输入'exit'退出）：")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "exit":
            break
        
        messages.append({"role": "user", "content": user_input})
        
        print("\n助手: ", end="")
        stream = client.chat(messages, stream=True)
        response = ""
        for chunk in stream:
            print(chunk, end="", flush=True)
            response += chunk
        print()
        
        messages.append({"role": "assistant", "content": response})


def test_provider_selection():
    """测试提供商选择"""
    print("\n" + "=" * 50)
    print("测试提供商选择")
    print("=" * 50)
    
    client = LLMClient("config.yaml")
    
    # 测试不同提供商
    providers = ["gpt-4", "gpt-3.5", "claude-3-opus", "claude-3-sonnet", "gemini-pro", "zhipu-chatglm"]
    
    for provider in providers:
        try:
            print(f"\n使用提供商: {provider}")
            prompt = "你好，请用一句话介绍自己"
            print(f"提示词: {prompt}")
            
            # 设置提供商
            client.proxy_pool.providers = {provider: client.proxy_pool.providers.get(provider)}
            if not client.proxy_pool.providers:
                print(f"提供商 {provider} 不可用")
                continue
            
            # 生成
            start_time = time.time()
            result = client.generate(prompt)
            end_time = time.time()
            print(f"结果: {result}")
            print(f"耗时: {end_time - start_time:.2f}秒")
        except Exception as e:
            print(f"错误: {str(e)}")


if __name__ == "__main__":
    # 测试生成功能
    test_generate()
    
    # 测试聊天功能
    test_chat()
    
    # 测试提供商选择
    test_provider_selection() 