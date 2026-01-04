# LLM系统

这是一个通用的大语言模型调用系统，支持流式和非流式调用，通过YAML配置支持多个SOTA LLM模型，并通过模型代理池自动选择调用SaaS LLM。

## 功能特点

- 支持多个主流LLM提供商：OpenAI、Anthropic、Google、智谱AI、阿里云通义千问、SiliconFlow等
- 支持流式和非流式调用
- 通过YAML配置文件灵活配置模型参数
- 模型代理池自动选择最佳模型
- 支持多种负载均衡策略：轮询、随机、优先级
- 支持故障转移和自动重试
- 健康检查和自动恢复
- 支持Qwen系列模型的特殊功能：思考过程、网络搜索、多模态等

## 支持的模型

### 阿里云通义千问系列
- **商业版模型**：qwen-max、qwen-plus、qwen-turbo系列
- **开源模型**：qwen3系列、qwen2.5系列、qwen2系列、qwen1.5系列
- **推理模型**：QwQ-32B-Preview、DeepSeek-R1系列
- **多模态模型**：qwen-vl系列、qwen2.5-vl系列
- **特殊功能**：支持思考过程(enable_thinking)、网络搜索(enable_search)

### SiliconFlow平台
- **Qwen系列**：Qwen2.5-72B-Instruct、Qwen2.5-7B-Instruct、Qwen3系列
- **推理模型**：QwQ-32B-Preview、DeepSeek-R1系列
- **特殊功能**：支持思考过程输出、top_k参数调节

### 其他平台
- OpenAI GPT系列
- Anthropic Claude系列
- Google Gemini系列
- 智谱AI GLM系列

## 安装

1. 克隆仓库：

```bash
git clone <repository-url>
cd llm_system
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 设置环境变量：

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export ZHIPU_API_KEY="your-zhipu-api-key"
export QWEN_API_KEY="your-qwen-api-key"
export SILICONFLOW_API_KEY="your-siliconflow-api-key"
```

## 配置

编辑`config.yaml`文件，配置模型参数和代理池设置。以下是一些关键配置示例：

### 阿里云通义千问配置

```yaml
models:
  # 商业版模型
  qwen-max:
    provider: qwen
    model: qwen-max
    api_key: ${QWEN_API_KEY}
    base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    max_tokens: 8192
    temperature: 0.7
    enable_thinking: true
    enable_search: true
    priority: 1

  # 开源模型
  qwen3-32b:
    provider: qwen
    model: qwen3-32b
    api_key: ${QWEN_API_KEY}
    base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    max_tokens: 32768
    temperature: 0.7
    priority: 2

  # 推理模型
  qwq-32b-preview:
    provider: qwen
    model: QwQ-32B-Preview
    api_key: ${QWEN_API_KEY}
    base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    max_tokens: 32768
    temperature: 0.7
    enable_thinking: true
    priority: 1
```

### SiliconFlow配置

```yaml
models:
  # SiliconFlow Qwen模型
  sf-qwen2.5-72b:
    provider: siliconflow
    model: Qwen/Qwen2.5-72B-Instruct
    api_key: ${SILICONFLOW_API_KEY}
    base_url: https://api.siliconflow.cn/v1
    max_tokens: 32768
    temperature: 0.7
    top_k: 50
    priority: 1

  # SiliconFlow推理模型
  sf-qwq-32b:
    provider: siliconflow
    model: Qwen/QwQ-32B-Preview
    api_key: ${SILICONFLOW_API_KEY}
    base_url: https://api.siliconflow.cn/v1
    max_tokens: 32768
    temperature: 0.7
    top_k: 50
    priority: 1
```

## 使用方法

### 命令行使用

1. 生成模式：

```bash
# 非流式
python main.py --mode generate --prompt "你好，请介绍一下自己"

# 流式
python main.py --mode generate --prompt "你好，请介绍一下自己" --stream

# 指定模型
python main.py --mode generate --prompt "请解释量子计算的原理" --model qwen-max
```

2. 聊天模式：

```bash
# 交互式聊天
python main.py --mode chat

# 从JSON文件加载消息
python main.py --mode chat --messages messages.json

# 流式聊天
python main.py --mode chat --stream

# 使用推理模型
python main.py --mode chat --model qwq-32b-preview
```

### 在代码中使用

```python
from main import LLMClient

# 初始化客户端
client = LLMClient("config.yaml")

# 生成文本
result = client.generate("你好，请介绍一下自己")
print(result)

# 流式生成
stream = client.generate("你好，请介绍一下自己", stream=True)
for chunk in stream:
    print(chunk, end="", flush=True)

# 聊天
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
    {"role": "user", "content": "请介绍一下自己"}
]
result = client.chat(messages)
print(result)

# 流式聊天
stream = client.chat(messages, stream=True)
for chunk in stream:
    print(chunk, end="", flush=True)

# 使用特定模型
result = client.generate("请解释深度学习的原理", model="qwen-max")
print(result)
```

### Qwen模型特殊功能

#### 思考过程模式
支持思考过程的模型（如qwen-max、qwq-32b-preview）会在响应中包含思考过程：

```python
# 使用思考模式
result = client.generate("请解决这个数学问题：2x + 3 = 7", model="qwq-32b-preview")
print(result)
# 输出会包含思考过程和最终答案
```

#### 网络搜索功能
部分模型支持网络搜索功能：

```python
# 使用网络搜索
result = client.generate("今天的天气如何？", model="qwen-max")
print(result)
# 模型会自动进行网络搜索并提供实时信息
```

## 测试

运行测试脚本验证所有Qwen模型功能：

```bash
# 运行所有测试
python test_qwen_models.py

# 测试特定平台
python test_qwen_models.py --platform alibaba
python test_qwen_models.py --platform siliconflow

# 测试特定模型类型
python test_qwen_models.py --model-type commercial
python test_qwen_models.py --model-type reasoning

# 详细测试报告
python test_qwen_models.py --verbose

# 测试特定功能
python test_qwen_models.py --test-type thinking
python test_qwen_models.py --test-type search
```

## 消息格式

聊天模式的消息格式为JSON数组，每个元素包含`role`和`content`字段：

```json
[
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
  {"role": "user", "content": "请介绍一下自己"}
]
```

## 模型选择建议

### 根据任务类型选择模型

1. **日常对话**：qwen-turbo、qwen2.5-7b-instruct
2. **复杂推理**：qwen-max、qwq-32b-preview、deepseek-r1
3. **代码生成**：qwen2.5-coder系列、qwen-max
4. **数学计算**：qwq-32b-preview、deepseek-r1-distill
5. **创意写作**：qwen-max、qwen-plus
6. **多模态任务**：qwen-vl系列、qwen2.5-vl系列

### 根据性能需求选择

1. **高性能**：qwen-max、qwen3-235b
2. **平衡性能**：qwen-plus、qwen2.5-72b
3. **快速响应**：qwen-turbo、qwen2.5-7b
4. **成本优化**：qwen2.5-1.5b、qwen2-0.5b

## 故障排除

### 常见问题

1. **API密钥错误**：确保环境变量设置正确
2. **网络连接问题**：检查网络连接和防火墙设置
3. **模型不可用**：某些模型可能需要申请权限
4. **思考过程不显示**：确保模型支持enable_thinking参数

### 调试模式

```bash
# 启用调试模式
python main.py --mode generate --prompt "测试" --debug

# 查看详细日志
python main.py --mode generate --prompt "测试" --log-level DEBUG
```

## 扩展

要添加新的LLM提供商，请按照以下步骤操作：

1. 在`models.py`中创建一个新的提供商类，继承自`LLMProvider`
2. 实现`generate`和`chat`方法
3. 在`LLMProxyPool._initialize_providers`方法中添加新提供商
4. 更新`config.yaml`配置文件

## 许可证

MIT 