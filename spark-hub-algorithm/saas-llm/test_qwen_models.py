#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen模型测试脚本
测试阿里云通义千问和硅基流动平台的所有Qwen系列模型
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from main import LLMClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qwen_test")

class QwenModelTester:
    """Qwen模型测试器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.client = LLMClient(config_path)
        self.test_results = {}
        
        # 测试用例
        self.test_cases = {
            "basic_chat": {
                "description": "基础对话测试",
                "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
                "expected_keywords": ["通义千问", "阿里", "AI", "助手", "模型"]
            },
            "math_reasoning": {
                "description": "数学推理测试",
                "messages": [{"role": "user", "content": "请计算：如果一个圆的半径是5厘米，那么它的面积是多少平方厘米？请详细说明计算过程。"}],
                "expected_keywords": ["π", "面积", "25π", "78.5", "公式"]
            },
            "code_generation": {
                "description": "代码生成测试",
                "messages": [{"role": "user", "content": "请用Python写一个函数，计算斐波那契数列的第n项"}],
                "expected_keywords": ["def", "fibonacci", "return", "python"]
            },
            "creative_writing": {
                "description": "创意写作测试",
                "messages": [{"role": "user", "content": "请写一首关于春天的短诗"}],
                "expected_keywords": ["春", "花", "绿", "暖", "生机"]
            },
            "thinking_test": {
                "description": "思考过程测试（仅支持思考模式的模型）",
                "messages": [{"role": "user", "content": "请分析一下为什么人工智能发展如此迅速？"}],
                "expected_keywords": ["人工智能", "发展", "技术", "数据"],
                "enable_thinking": True
            },
            "search_test": {
                "description": "联网搜索测试（仅支持搜索的模型）",
                "messages": [{"role": "user", "content": "今天的天气怎么样？"}],
                "expected_keywords": ["天气", "温度", "今天"],
                "enable_search": True
            }
        }
        
        # 按类别分组的模型列表
        self.model_groups = {
            "阿里云商业版": [
                "qwen-max", "qwen-max-latest", "qwen-max-2025-01-25",
                "qwen-plus", "qwen-plus-latest", "qwen-plus-2025-04-28", "qwen-plus-2025-01-25",
                "qwen-turbo", "qwen-turbo-latest", "qwen-turbo-2025-04-28", "qwen-turbo-2024-11-01",
                "qwq-plus", "qvq-max", "qvq-max-latest",
                "qwen-vl-max", "qwen-vl-plus"
            ],
            "阿里云开源版": [
                "qwen3-235b-a22b", "qwen3-32b", "qwen3-30b-a3b", "qwen3-14b", "qwen3-8b", "qwen3-4b", "qwen3-1.7b", "qwen3-0.6b",
                "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct",
                "qwen2.5-14b-instruct-1m", "qwen2.5-7b-instruct-1m",
                "qwen2-72b-instruct", "qwen2-57b-a14b-instruct", "qwen2-7b-instruct",
                "qwen1.5-110b-chat", "qwen1.5-72b-chat", "qwen1.5-32b-chat", "qwen1.5-14b-chat", "qwen1.5-7b-chat",
                "qwen2.5-vl-72b-instruct", "qwen2.5-vl-7b-instruct", "qwen2.5-vl-3b-instruct",
                "qwq-32b"
            ],
            "DeepSeek-R1系列": [
                "deepseek-r1", "deepseek-r1-0528",
                "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-qwen-7b", "deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-qwen-32b",
                "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-llama-70b"
            ],
            "硅基流动Qwen系列": [
                "sf-qwen2.5-72b-instruct", "sf-qwen2.5-32b-instruct", "sf-qwen2.5-14b-instruct", "sf-qwen2.5-7b-instruct",
                "sf-qwen2.5-3b-instruct", "sf-qwen2.5-1.5b-instruct", "sf-qwen2.5-0.5b-instruct",
                "sf-qwen3-235b-a22b", "sf-qwen3-32b",
                "sf-qwq-32b-preview"
            ],
            "硅基流动DeepSeek-R1": [
                "sf-deepseek-r1",
                "sf-deepseek-r1-distill-qwen-1.5b", "sf-deepseek-r1-distill-qwen-7b", "sf-deepseek-r1-distill-qwen-14b", "sf-deepseek-r1-distill-qwen-32b",
                "sf-deepseek-r1-distill-llama-8b", "sf-deepseek-r1-distill-llama-70b"
            ]
        }
        
        # 支持特殊功能的模型
        self.thinking_models = [
            "qwen-plus", "qwen-plus-latest", "qwen-plus-2025-04-28",
            "qwen-turbo-latest", "qwen-turbo-2025-04-28",
            "qwq-plus", "qvq-max", "qvq-max-latest",
            "qwen3-235b-a22b", "qwen3-32b", "qwen3-30b-a3b", "qwen3-14b", "qwen3-8b", "qwen3-4b", "qwen3-1.7b", "qwen3-0.6b",
            "qwq-32b"
        ] + [model for model in sum(self.model_groups.values(), []) if "deepseek-r1" in model]
        
        self.search_models = [
            "qwen-plus", "qwen-plus-latest", "qwen-plus-2025-04-28"
        ]
    
    def test_model(self, model_name: str, test_case: str, **kwargs) -> Dict[str, Any]:
        """测试单个模型"""
        test_info = self.test_cases[test_case]
        logger.info(f"测试模型 {model_name} - {test_info['description']}")
        
        start_time = time.time()
        try:
            # 检查特殊功能支持
            if test_case == "thinking_test" and model_name not in self.thinking_models:
                return {
                    "status": "skipped",
                    "reason": "模型不支持思考模式",
                    "response": None,
                    "duration": 0,
                    "error": None
                }
            
            if test_case == "search_test" and model_name not in self.search_models:
                return {
                    "status": "skipped",
                    "reason": "模型不支持联网搜索",
                    "response": None,
                    "duration": 0,
                    "error": None
                }
            
            # 准备参数
            test_kwargs = kwargs.copy()
            if test_info.get("enable_thinking"):
                test_kwargs["enable_thinking"] = True
            if test_info.get("enable_search"):
                test_kwargs["enable_search"] = True
            
            # 临时修改配置选择特定模型
            original_strategy = self.client.proxy_pool.strategy
            original_providers = self.client.proxy_pool.providers.copy()
            
            # 只保留目标模型
            if model_name in self.client.proxy_pool.providers:
                self.client.proxy_pool.providers = {model_name: self.client.proxy_pool.providers[model_name]}
                self.client.proxy_pool.strategy = "priority"
                
                # 执行测试
                response = self.client.chat(
                    test_info["messages"], 
                    stream=False,
                    max_tokens=1000,
                    temperature=0.7,
                    **test_kwargs
                )
                
                duration = time.time() - start_time
                
                # 检查响应质量
                quality_score = self._evaluate_response_quality(response, test_info.get("expected_keywords", []))
                
                return {
                    "status": "success",
                    "response": response,
                    "duration": duration,
                    "quality_score": quality_score,
                    "error": None
                }
            else:
                return {
                    "status": "error",
                    "reason": f"模型 {model_name} 未在配置中找到",
                    "response": None,
                    "duration": 0,
                    "error": "Model not found"
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"测试模型 {model_name} 失败: {str(e)}")
            return {
                "status": "error",
                "response": None,
                "duration": duration,
                "error": str(e)
            }
        finally:
            # 恢复原始配置
            self.client.proxy_pool.strategy = original_strategy
            self.client.proxy_pool.providers = original_providers
    
    def _evaluate_response_quality(self, response: str, expected_keywords: List[str]) -> float:
        """评估响应质量"""
        if not response or not expected_keywords:
            return 0.0
        
        response_lower = response.lower()
        matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        
        # 基础关键词匹配分数
        keyword_score = matched_keywords / len(expected_keywords)
        
        # 响应长度分数（适中长度得分更高）
        length_score = min(len(response) / 200, 1.0) if len(response) > 50 else len(response) / 50
        
        # 综合分数
        return (keyword_score * 0.7 + length_score * 0.3)
    
    def test_model_group(self, group_name: str, test_cases: List[str] = None) -> Dict[str, Any]:
        """测试模型组"""
        if group_name not in self.model_groups:
            raise ValueError(f"未知模型组: {group_name}")
        
        if test_cases is None:
            test_cases = ["basic_chat", "math_reasoning", "code_generation"]
        
        models = self.model_groups[group_name]
        group_results = {}
        
        logger.info(f"开始测试模型组: {group_name}")
        logger.info(f"包含模型: {', '.join(models)}")
        
        for model_name in models:
            model_results = {}
            for test_case in test_cases:
                result = self.test_model(model_name, test_case)
                model_results[test_case] = result
                
                # 简短的结果输出
                if result["status"] == "success":
                    logger.info(f"✓ {model_name} - {test_case}: 成功 (质量分数: {result['quality_score']:.2f}, 耗时: {result['duration']:.2f}s)")
                elif result["status"] == "skipped":
                    logger.info(f"⚬ {model_name} - {test_case}: 跳过 ({result['reason']})")
                else:
                    logger.error(f"✗ {model_name} - {test_case}: 失败 ({result['error']})")
                
                # 添加延迟避免API限制
                time.sleep(1)
            
            group_results[model_name] = model_results
        
        return group_results
    
    def test_all_models(self, test_cases: List[str] = None) -> Dict[str, Any]:
        """测试所有模型"""
        if test_cases is None:
            test_cases = ["basic_chat", "math_reasoning"]
        
        all_results = {}
        
        for group_name in self.model_groups:
            try:
                group_results = self.test_model_group(group_name, test_cases)
                all_results[group_name] = group_results
            except Exception as e:
                logger.error(f"测试模型组 {group_name} 失败: {str(e)}")
                all_results[group_name] = {"error": str(e)}
        
        return all_results
    
    def test_specific_models(self, model_names: List[str], test_cases: List[str] = None) -> Dict[str, Any]:
        """测试指定模型"""
        if test_cases is None:
            test_cases = ["basic_chat"]
        
        results = {}
        
        for model_name in model_names:
            model_results = {}
            for test_case in test_cases:
                result = self.test_model(model_name, test_case)
                model_results[test_case] = result
                
                # 显示详细结果
                if result["status"] == "success":
                    logger.info(f"✓ {model_name} - {test_case}: 成功")
                    logger.info(f"  响应: {result['response'][:200]}...")
                    logger.info(f"  质量分数: {result['quality_score']:.2f}")
                    logger.info(f"  耗时: {result['duration']:.2f}s")
                elif result["status"] == "skipped":
                    logger.info(f"⚬ {model_name} - {test_case}: 跳过 ({result['reason']})")
                else:
                    logger.error(f"✗ {model_name} - {test_case}: 失败 ({result['error']})")
                
                time.sleep(1)
            
            results[model_name] = model_results
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """生成测试报告"""
        report_lines = []
        report_lines.append("# Qwen模型测试报告")
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for group_name, group_results in results.items():
            if isinstance(group_results, dict) and "error" in group_results:
                report_lines.append(f"## {group_name}")
                report_lines.append(f"❌ 组测试失败: {group_results['error']}")
                report_lines.append("")
                continue
            
            report_lines.append(f"## {group_name}")
            report_lines.append("")
            
            for model_name, model_results in group_results.items():
                report_lines.append(f"### {model_name}")
                
                for test_case, result in model_results.items():
                    total_tests += 1
                    
                    if result["status"] == "success":
                        successful_tests += 1
                        status_icon = "✅"
                        details = f"质量分数: {result['quality_score']:.2f}, 耗时: {result['duration']:.2f}s"
                    elif result["status"] == "skipped":
                        skipped_tests += 1
                        status_icon = "⚬"
                        details = f"跳过原因: {result.get('reason', 'N/A')}"
                    else:
                        failed_tests += 1
                        status_icon = "❌"
                        details = f"错误: {result.get('error', 'Unknown error')}"
                    
                    report_lines.append(f"- {status_icon} **{test_case}**: {details}")
                
                report_lines.append("")
        
        # 添加统计摘要
        report_lines.append("## 测试统计")
        report_lines.append(f"- 总测试数: {total_tests}")
        report_lines.append(f"- 成功: {successful_tests}")
        report_lines.append(f"- 失败: {failed_tests}")
        report_lines.append(f"- 跳过: {skipped_tests}")
        report_lines.append(f"- 成功率: {(successful_tests / total_tests * 100):.1f}%")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"测试报告已保存到: {output_file}")
        
        return report_content


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen模型测试工具")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--group", type=str, help="测试特定模型组")
    parser.add_argument("--models", type=str, nargs="+", help="测试特定模型")
    parser.add_argument("--test-cases", type=str, nargs="+", 
                       choices=["basic_chat", "math_reasoning", "code_generation", "creative_writing", "thinking_test", "search_test"],
                       default=["basic_chat", "math_reasoning"],
                       help="要执行的测试用例")
    parser.add_argument("--output", type=str, help="输出报告文件路径")
    parser.add_argument("--all", action="store_true", help="测试所有模型")
    parser.add_argument("--quick", action="store_true", help="快速测试（仅基础对话）")
    parser.add_argument("--list-groups", action="store_true", help="列出所有模型组")
    parser.add_argument("--list-models", action="store_true", help="列出所有模型")
    
    args = parser.parse_args()
    
    # 检查环境变量
    required_env_vars = ["DASHSCOPE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"缺少必要的环境变量: {', '.join(missing_vars)}")
        logger.error("请设置DASHSCOPE_API_KEY环境变量")
        sys.exit(1)
    
    tester = QwenModelTester(args.config)
    
    if args.list_groups:
        print("可用的模型组:")
        for group_name, models in tester.model_groups.items():
            print(f"  {group_name}: {len(models)}个模型")
        return
    
    if args.list_models:
        print("所有可用模型:")
        for group_name, models in tester.model_groups.items():
            print(f"\n{group_name}:")
            for model in models:
                print(f"  - {model}")
        return
    
    # 确定测试用例
    test_cases = ["basic_chat"] if args.quick else args.test_cases
    
    # 执行测试
    results = None
    
    if args.all:
        logger.info("开始测试所有模型...")
        results = tester.test_all_models(test_cases)
    elif args.group:
        logger.info(f"开始测试模型组: {args.group}")
        if args.group not in tester.model_groups:
            logger.error(f"未知模型组: {args.group}")
            logger.error(f"可用组: {', '.join(tester.model_groups.keys())}")
            sys.exit(1)
        results = {args.group: tester.test_model_group(args.group, test_cases)}
    elif args.models:
        logger.info(f"开始测试指定模型: {', '.join(args.models)}")
        results = {"指定模型": tester.test_specific_models(args.models, test_cases)}
    else:
        # 默认测试几个代表性模型
        representative_models = ["qwen-plus", "qwen2.5-72b-instruct", "sf-qwen2.5-7b-instruct"]
        logger.info(f"开始测试代表性模型: {', '.join(representative_models)}")
        results = {"代表性模型": tester.test_specific_models(representative_models, test_cases)}
    
    # 生成报告
    if results:
        report = tester.generate_report(results, args.output)
        if not args.output:
            print("\n" + "="*50)
            print(report)


if __name__ == "__main__":
    main() 