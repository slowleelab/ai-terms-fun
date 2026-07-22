---
title: Function Calling 函数调用
slug: function-calling
category: 进阶专题
tags: [Function Calling, Tool Use, 结构化输出, Agent, OpenAI]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Function Calling 函数调用

> 五层读懂一个词。这次拆的是：**Function Calling**--LLM 工具调用的工业标准。从 ReAct 的「文本解析 Action」升级为「结构化 JSON 输出」，模型直接吐 function name + arguments，稳定、可解析、可并行。OpenAI 2023-06 推出后成为 Agent 工具调用的事实标准。

---

## L1 · 一句话点破

**Function Calling = LLM 结构化工具调用接口**。模型根据工具 schema 直接输出 `{name, arguments}` 的 JSON，不再需要解析「Action: Search[...]」之类的自由文本。一次可并行调用多个函数，支持工具选择强制（`tool_choice`）。是 ReAct 的工程化升级，现代 Agent 工具调用的基础。

---

## L2 · 通俗类比

ReAct 的工具调用像**写信下指令**：

```
Thought: 我要查天气
Action: Search[北京今天天气]
```

- 模型按格式输出文本
- 程序用正则解析 `Search[...]`
- 格式错了（漏了括号、多了空格），解析失败
- 一次只能调一个工具

**Function Calling 像填表**：

```json
{"name": "get_weather", "arguments": {"city": "北京", "date": "today"}}
```

- 模型直接吐 JSON
- 程序直接 `json.loads`
- 模型被训练成按 schema 输出，几乎不会格式错
- 一次可填多张表（并行调用）

**对比**：

| 维度 | ReAct 文本格式 | Function Calling |
|------|---------------|------------------|
| 输出格式 | 自由文本 | 结构化 JSON |
| 解析 | 正则，脆弱 | `json.loads`，稳健 |
| 参数类型 | 字符串 | 强类型（string/number/bool/array/object） |
| 并行调用 | ❌（串行） | ✅（一次多个） |
| 错误率 | 高（格式错） | 低（训练保证） |
| 工具选择 | 模型自选 | 可强制（`tool_choice`） |

**关键能力**：

- **结构化输出**：JSON schema 约束参数类型和必填字段
- **并行调用**：一次输出多个 function call，并行执行
- **强制选择**：`tool_choice` 指定必须调某函数或必须调一个
- **流式输出**：支持流式返回 function call

**代价**：

- 需要模型支持（OpenAI / Anthropic / 开源部分支持）
- 工具描述消耗 token（schema 越复杂消耗越大）
- 复杂参数（嵌套对象）模型可能填错
- 不是所有任务都适合结构化（创意任务不需要）

**适用**：

- Agent 工具调用（事实标准）
- 结构化数据提取（从文本提取 JSON）
- 多工具编排
- 与外部 API 集成

---

## L3 · 正经定义

**Function Calling**：LLM API 接口能力，允许开发者向模型注册一组函数（含 name/description/parameters schema），模型在回复时根据需要输出结构化的 function call（含 function name 和 arguments JSON），而非自由文本。开发者执行函数后将结果返回模型，模型据此继续推理或生成最终回复。

**OpenAI Function Calling 流程**：

```
1. 开发者定义函数 schema:
   {
     "name": "get_weather",
     "description": "获取指定城市天气",
     "parameters": {
       "type": "object",
       "properties": {
         "city": {"type": "string", "description": "城市名"},
         "date": {"type": "string", "enum": ["today", "tomorrow"]}
       },
       "required": ["city"]
     }
   }

2. 调用 API 时传入 tools=[...] 和 tool_choice

3. 模型回复:
   - 不需要工具: 正常文本回复
   - 需要工具: 返回 tool_calls = [
       {"id": "call_xxx", "name": "get_weather", "arguments": '{"city":"北京","date":"today"}'}
     ]

4. 开发者执行 get_weather("北京", "today")，得到结果

5. 把结果作为 role="tool" 的消息追加，再次调用 API

6. 模型根据工具结果生成最终回复
```

**关键特性**：

- **schema 约束**：JSON Schema 定义参数类型、必填、枚举
- **并行调用**：一次可返回多个 tool_calls
- **强制选择**：`tool_choice: {"type": "function", "function": {"name": "xxx"}}`
- **流式支持**：支持 stream 模式
- **多轮对话**：工具结果可加入对话历史

**参考资料**：

- 📝 OpenAI Function Calling 文档：https://platform.openai.com/docs/guides/function-calling
- 📝 Anthropic Tool Use：https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- 📄 Patil et al., *Gorilla: Large Language Model Connected with Massive APIs*, 2023
- 📄 Tang et al., *ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs*, 2023
- 🔧 LangChain Tools：https://python.langchain.com/docs/modules/tools/

---

## L4 · 原理深挖

### 4.1 从 ReAct 到 Function Calling

**ReAct 的问题**：

```
模型输出:
Thought: 我要查北京天气
Action: Search[北京今天天气]

解析:
- 正则: r'Action: (\w+)\[(.+)\]'
- 失败情况:
  - Action: search("北京今天天气")  # 格式不对
  - Action: Search[北京, 今天]      # 参数有逗号，正则错
  - Action: Search 北京今天天气      # 漏括号
```

**Function Calling 的解决**：

- 模型被训练成直接输出 JSON
- 不需要解析自由文本
- schema 约束保证参数类型对

**训练方式**：

- SFT：用大量 function calling 数据微调
- RLHF：奖励正确的 function call
- 模型学会按 schema 输出 JSON

### 4.2 Function Calling 的 API

**OpenAI Function Calling**：

```python
from openai import OpenAI
client = OpenAI()

# 1. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市指定日期的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "date": {"type": "string", "enum": ["today", "tomorrow"]}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    }
]

# 2. 第一次调用，模型决定是否调用工具
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎样？25 度的话穿什么？"}],
    tools=tools,
    tool_choice="auto",  # auto/none/required/指定函数
)

# 3. 模型返回 tool_calls
# response.choices[0].message.tool_calls = [
#   {"id": "call_1", "name": "get_weather", "arguments": '{"city":"北京","date":"today"}'}
# ]

# 4. 执行工具
import json
tool_calls = response.choices[0].message.tool_calls
messages = [response.choices[0].message]

for call in tool_calls:
    args = json.loads(call.function.arguments)
    if call.function.name == "get_weather":
        result = get_weather(**args)  # 假设已实现
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result),
    })

# 5. 第二次调用，模型根据工具结果生成最终回复
final = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
)
print(final.choices[0].message.content)
# "北京今天 25 度，建议穿薄长袖..."
```

### 4.3 并行 Function Calling

**OpenAI 2023-11 起支持并行调用**：

```python
# 用户问题需要多个工具
# "北京和上海今天天气怎样？"
# 模型一次返回多个 tool_calls:
# [
#   {"id": "call_1", "name": "get_weather", "arguments": '{"city":"北京","date":"today"}'},
#   {"id": "call_2", "name": "get_weather", "arguments": '{"city":"上海","date":"today"}'}
# ]

# 开发者并行执行:
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(execute_tool, call): call
        for call in tool_calls
    }
    for future in concurrent.futures.as_completed(futures):
        call = futures[future]
        result = future.result()
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
```

**优势**：

- 延迟降低（并行 vs 串行）
- 适合多源数据查询
- 减少多轮对话

### 4.4 tool_choice 详解

**`tool_choice` 选项**：

| 值 | 含义 |
|----|------|
| `"auto"`（默认） | 模型自主决定是否调用工具 |
| `"none"` | 禁止调用工具，强制文本回复 |
| `"required"` | 必须调用至少一个工具 |
| `{"type": "function", "function": {"name": "xxx"}}` | 强制调用指定函数 |

**使用场景**：

```python
# 场景 1: 用户闲聊，不需要工具
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}],
    tools=tools,
    tool_choice="none",  # 禁止工具
)

# 场景 2: 强制结构化输出
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "提取这段文本的事件"}],
    tools=[extract_event_tool],
    tool_choice={"type": "function", "function": {"name": "extract_event"}},
    # 强制调用，保证输出结构化
)

# 场景 3: Agent 自主决策
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # 模型决定
)
```

### 4.5 JSON Schema 与参数约束

**复杂 schema 示例**：

```python
{
    "name": "book_flight",
    "description": "预订机票",
    "parameters": {
        "type": "object",
        "properties": {
            "from": {"type": "string", "description": "出发城市"},
            "to": {"type": "string", "description": "目的城市"},
            "date": {"type": "string", "format": "date", "description": "日期 YYYY-MM-DD"},
            "passengers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer", "minimum": 0},
                        "type": {"type": "string", "enum": ["adult", "child", "infant"]}
                    },
                    "required": ["name", "type"]
                },
                "minItems": 1
            },
            "cabin": {"type": "string", "enum": ["economy", "business", "first"], "default": "economy"}
        },
        "required": ["from", "to", "date", "passengers"]
    }
}
```

**模型输出**：

```json
{
    "name": "book_flight",
    "arguments": {
        "from": "北京",
        "to": "上海",
        "date": "2026-08-01",
        "passengers": [
            {"name": "张三", "age": 30, "type": "adult"},
            {"name": "李四", "age": 8, "type": "child"}
        ],
        "cabin": "economy"
    }
}
```

**schema 设计要点**：

- **description 清晰**：模型靠 description 理解参数含义
- **enum 约束**：固定取值用 enum
- **required 明确**：必填字段标 required
- **避免过深嵌套**：3 层以内模型表现好
- **format 提示**：date-time/date/uuid 等格式用 format

### 4.6 Function Calling 的实现原理

**模型训练**：

1. **SFT 阶段**：用大量 (prompt, tool_calls, tool_results, response) 数据微调
2. **RLHF 阶段**：奖励正确的 function call、惩罚格式错或参数错
3. **特殊 token**：训练时用特殊 token 标记 function call 边界

**推理时**：

- 工具 schema 作为系统 prompt 的一部分
- 模型生成时遇到需要工具，输出特殊格式的 function call
- 框架解析并执行

**开源模型的支持**：

| 模型 | Function Calling 支持 |
|------|----------------------|
| GPT-4 / GPT-3.5 | ✅（原生） |
| Claude 3 | ✅（Tool Use） |
| Gemini | ✅ |
| Llama 3.1 | ✅（训练支持） |
| Qwen 2.5 | ✅ |
| Mistral | ✅（通过 fine-tune） |
| 早期开源模型 | ⚠️（需 prompt 工程） |

### 4.7 Function Calling 的应用

**应用 1: Agent 工具调用**

```python
# ReAct + Function Calling = 现代 Agent
def agent_loop(question, tools, llm, max_turns=10):
    messages = [{"role": "user", "content": question}]
    
    for _ in range(max_turns):
        response = llm(messages=messages, tools=tools, tool_choice="auto")
        msg = response.choices[0].message
        
        if not msg.tool_calls:
            return msg.content  # 最终回复
        
        messages.append(msg)
        for call in msg.tool_calls:
            result = execute_tool(call)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })
    
    return "达到最大轮数"
```

**应用 2: 结构化数据提取**

```python
# 从非结构化文本提取结构化数据
tools = [{
    "type": "function",
    "function": {
        "name": "extract_event",
        "description": "从文本提取事件信息",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "format": "date"},
                "location": {"type": "string"},
                "participants": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title", "date"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "2026 年 8 月 15 日，张三和李四在北京参加 AI 大会"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_event"}},
)
# 模型强制输出结构化 event，无需正则解析
```

**应用 3: 多 API 编排**

```python
# 复杂任务: "查北京到上海的机票，再查上海酒店"
# 模型一次输出多个 tool_calls:
# - search_flights(from="北京", to="上海", date="...")
# - search_hotels(city="上海", checkin="...", checkout="...")
# 并行执行，合并结果
```

### 4.8 Function Calling 的局限

**局限 1: schema 复杂度限制**

- 嵌套层级太深（>3 层），模型容易填错
- 复杂 union type 支持差
- 大量可选字段模型可能漏填或乱填

**局限 2: 参数值幻觉**

- 模型可能编造不存在的参数值
- 如日期幻觉、ID 编造
- 需要工具层做参数校验

**局限 3: token 消耗**

- 工具 schema 占用大量 token
- 工具多时（>20 个）token 爆炸
- 需要工具检索（RAG over tools）

**局限 4: 不是所有模型支持**

- 老模型、小模型支持差
- 开源模型需要 fine-tune
- 不同厂商 API 不完全兼容

**局限 5: 工具选择错误**

- 模型可能选错工具
- 特别是工具描述相似时
- 需要 few-shot 示例或工具分组

**局限 6: 多轮工具调用上下文膨胀**

- 每轮 tool_call + tool_result 都进对话历史
- 多轮后上下文爆炸
- 需要历史压缩或总结

**局限 7: 流式输出复杂**

- 流式时 function call 分多个 chunk
- 需要累积解析
- 错误处理复杂

**局限 8: 错误处理**

- 工具执行失败，模型如何处理？
- 需要把错误信息返回给模型让其重试
- 但模型可能重复调用失败的工具

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-03**：OpenAI 推出 Chat Completions API，初版 function calling
- **2023-06**：OpenAI 正式发布 Function Calling（gpt-3.5-turbo-0613, gpt-4-0613）
- **2023-08**：Anthropic Claude 推出 Tool Use
- **2023-11**：OpenAI 支持并行 function calling
- **2024-01**：Google Gemini 支持 Function Calling
- **2024 下半年**：开源模型（Llama 3.1, Qwen 2.5）原生支持
- **2024-2025**：Function Calling 成为 Agent 工具调用事实标准，工具检索（RAG over tools）出现

### 5.2 常见坑

**坑 1: schema 描述不清**。description 写得太简单，模型不知道何时用、怎么用。要详细描述用途、输入、输出、边界。

**坑 2: 嵌套太深**。3 层以上嵌套模型容易填错。要扁平化或拆分多个函数。

**坑 3: 工具太多**。一次注册 50+ 工具，token 爆炸 + 模型选择困难。要工具检索（RAG over tools）或分组。

**坑 4: 没做参数校验**。模型可能填错参数值（如不存在的城市）。要在工具层校验。

**坑 5: 忽略 tool_call_id**。多轮对话时 tool result 必须带 tool_call_id，否则 API 报错。

**坑 6: 工具失败不返回错误信息**。工具执行失败直接抛异常，模型不知道。要把错误信息作为 tool result 返回。

**坑 7: 期望模型自动重试**。工具失败后，模型可能换个工具或换参数，但也可能死循环。要设最大重试次数。

**坑 8: 流式解析错**。流式时 function call 分多个 chunk，要累积解析。直接解析单个 chunk 会失败。

**坑 9: tool_choice 用错**。auto 模式下模型可能不调用工具，需要结构化输出时要用 required 或指定函数。

**坑 10: 忽略成本**。工具 schema 占 token，复杂 schema 一次调用几千 token。要精简 schema。

**坑 11: 多模型兼容问题**。不同厂商 function calling API 略有差异。要抽象层适配。

**坑 12: 没处理 function call 终止**。模型可能反复调用工具不收敛。要设最大轮数。

**坑 13: 期望 function calling 替代 ReAct**。Function Calling 是工具调用接口，ReAct 是推理+行动范式。现代 Agent 通常是 ReAct + Function Calling（推理用 Thought，工具用 function call）。

### 5.3 面试怎么考

1. **Function Calling 解决什么问题？** 答：ReAct 的文本格式 Action 解析脆弱、不能并行、不能约束参数类型。Function Calling 让模型直接输出结构化 JSON，按 schema 约束，可并行，可强制选择。
2. **Function Calling 的工作流程？** 答：定义工具 schema -> API 调用传入 tools -> 模型返回 tool_calls -> 执行工具 -> 把结果作为 role=tool 消息追加 -> 再次调用 API -> 模型生成最终回复。
3. **tool_choice 的几种模式？** 答：auto（模型自决）、none（禁用）、required（必须调一个）、指定函数（强制调某函数，用于结构化输出）。
4. **Function Calling 和 ReAct 的关系？** 答：Function Calling 是工具调用接口（怎么调），ReAct 是推理+行动范式（怎么想+怎么动）。现代 Agent 通常是 ReAct + Function Calling：Thought 推理用文本，Action 用 function call。
5. **Function Calling 的局限？** 答：schema 复杂度限制、参数值幻觉、token 消耗、模型支持差异、多轮上下文膨胀、错误处理复杂。

---

## 速记卡

**工作流程**：

```
1. 定义 tools schema (name/description/parameters)
2. 调用 API: messages + tools + tool_choice
3. 模型返回: tool_calls = [{id, name, arguments(JSON)}]
4. 执行工具，得到结果
5. 追加 role=tool 消息 (带 tool_call_id)
6. 再次调用 API
7. 模型生成最终回复 (或继续 tool_calls)
```

**tool_choice 选项**：

| 值 | 含义 |
|----|------|
| `"auto"` | 模型自决 |
| `"none"` | 禁用 |
| `"required"` | 必调一个 |
| 指定函数 | 强制调某函数 |

**schema 设计要点**：

| 要点 | 说明 |
|------|------|
| description 清晰 | 模型靠 description 理解 |
| enum 约束 | 固定取值 |
| required 明确 | 必填字段 |
| 嵌套 ≤3 层 | 太深易错 |
| format 提示 | date/uuid 等 |

**主流模型支持**：

| 模型 | 支持 |
|------|------|
| GPT-4/3.5 | ✅ 原生 |
| Claude 3 | ✅ Tool Use |
| Gemini | ✅ |
| Llama 3.1 | ✅ |
| Qwen 2.5 | ✅ |
| 早期开源 | ⚠️ 需 prompt 工程 |

**vs ReAct**：

| 维度 | ReAct | Function Calling |
|------|-------|------------------|
| 输出 | 自由文本 | JSON |
| 解析 | 正则 | `json.loads` |
| 并行 | ❌ | ✅ |
| 类型约束 | ❌ | ✅ |
| 强制选择 | ❌ | ✅ |

**一句话记忆**：Function Calling = LLM 结构化工具调用接口。模型按 JSON Schema 直接输出 `{name, arguments}`，无需解析自由文本，可并行调用、可强制选择（tool_choice）、有类型约束。是 ReAct 文本格式的工程化升级，现代 Agent 的事实标准。工作流：定义 schema -> 调 API 拿 tool_calls -> 执行 -> 结果作为 role=tool 追加 -> 再调 API。局限：schema 复杂度、参数幻觉、token 消耗、模型兼容。现代 Agent 通常是 ReAct + Function Calling（Thought 推理 + function call 行动）。

---

> *上一篇：[ReAct 推理与行动](./react) -- ReAct 用文本格式 Action，Function Calling 把它升级为结构化 JSON。*
> *下一篇：[Planning 任务规划](./planning) -- ReAct 边想边做，Planning 先想好再做，适合复杂任务。*
