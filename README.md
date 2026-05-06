# claude-cli → OpenAI 兼容 API

把本地登录的 `claude` CLI 包成 OpenAI 兼容协议的 HTTP 服务，支持并发、重试、流式。

> English version: [README.en.md](./README.en.md)

## 快速开始

```bash
pip install -r requirements.txt
python claude_api.py            # 起服务: http://127.0.0.1:8765
python parallel_client_demo.py  # 另开终端, 跑并行示例
python multi_turn_demo.py       # 多轮对话示例
```

OpenAI SDK 直接指过去：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-used")
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "hello"}],
)
print(r.choices[0].message.content)
```

---

## 模型名称怎么选

`claude` CLI 的 `--model` 参数支持几种"别名"和完整 ID。直接抄 [官方文档](https://code.claude.com/docs/en/model-config) 的表：

| Model alias | 行为 |
|---|---|
| `default` | 特殊值，清除任何 model 覆盖，回到账号类型对应的推荐模型（本身不是模型别名） |
| `best` | 当前最强模型，目前等于 `opus` |
| `sonnet` | 最新 Sonnet，适合日常 coding |
| `opus` | 最新 Opus，适合复杂推理 |
| `haiku` | 最新 Haiku，快、便宜，适合简单任务 |
| `sonnet[1m]` | Sonnet 1M 上下文窗口，适合长对话/长文档 |
| `opus[1m]` | Opus 1M 上下文窗口 |
| `opusplan` | 计划阶段用 opus，执行阶段切到 sonnet |

也可以传**完整 ID**（如 `claude-sonnet-4-6`）钉死某个具体版本。这些值都直接透传给 `claude --model`。

### 服务端的额外约定

为了配合 OpenAI 客户端必填 `model` 字段的硬性要求，本服务对几个"占位字符串"做了识别：

- `default` / `auto` / `claude` / `claude-cli` / 空字符串 → **不**给 CLI 加 `--model`，直接走账号默认（和官方 `default` 别名行为一致）
- 其它任何字符串原样透传给 `claude --model`，由 CLI 校验

实操建议：

- **不知道选啥**：填 `default`，跟着账号走
- **想要最快**：填 `haiku`
- **想要最强**：填 `opus` 或 `best`
- **长对话/塞大段文档**：填 `sonnet[1m]`
- **想钉死版本不被悄悄升级**：填完整 ID，比如 `claude-sonnet-4-6`

确认你这台机器上的 CLI 真的支持某个值，可以直接试一下：

```bash
echo "ping" | claude -p --model sonnet --output-format json | head
# 看返回里 "is_error" 是 false 还是 true
```

或者环境变量配个全局默认，客户端就可以一直传 `default`：

```bash
CLAUDE_DEFAULT_MODEL=sonnet[1m] python claude_api.py
```

---

## 多轮对话怎么做

有两种风格，**推荐第一种**（和 OpenAI API 完全一致）。

### 方式 1（推荐）：客户端维护历史，每次发完整 messages

服务端是无状态的——它把整个 `messages` 数组拼成一段 `User: ... / Assistant: ...` 对话送给 CLI。所以你像用 OpenAI 那样在 client 端追加历史就行：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="x")

history = [{"role": "system", "content": "你是一个简洁的中文助手。"}]

def chat(user_msg: str) -> str:
    history.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(model="default", messages=history)
    reply = r.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

print(chat("我叫张三"))
print(chat("我刚才说我叫什么?"))   # 模型能正确回忆
```

优点：

- 协议干净，所有 OpenAI 客户端 / 框架（LangChain、LlamaIndex、open-webui、LobeChat 等）天然兼容
- 服务端无状态，并发友好
- 历史完全在你手里，想剪枝、想分叉、想注入工具结果都方便

代价：每次都要把历史发一遍，长对话 token 用量高。如果聊得很久，可以做个 sliding window（保留最近 N 轮 + system）或者总结早期消息再放回 history。

完整可运行示例见 [`multi_turn_demo.py`](./multi_turn_demo.py)。

### 方式 2：传 `session_id` 复用 CLI 会话（享受 prompt cache）

服务端支持一个**可选扩展字段** `session_id`。传了它就走 CLI 原生 `--session-id` / `--resume` 路径——历史由 CLI 端维护，每次只把最后一条 user 消息送过去，**前缀稳定命中 prompt cache**，长对话省 token 也省时。

OpenAI Python SDK 用 `extra_body` 透传该字段（标准 OpenAI 客户端不会感知到，协议保持兼容）：

```python
import uuid
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="x")

SID = str(uuid.uuid4())   # 一段对话用一个

def chat(user_msg: str) -> str:
    r = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": user_msg}],
        extra_body={"session_id": SID},        # <-- 关键
    )
    return r.choices[0].message.content

print(chat("我叫张三, 在深圳做后端"))
print(chat("我刚说我在哪做什么?"))           # CLI 端自动续上, 享受缓存
```

完整示例：[`session_demo.py`](./session_demo.py)（带 `cache_read_input_tokens` / `cache_creation_input_tokens` 打印，可以直接看到缓存收益）。

#### 服务端的处理逻辑

- 收到带 `session_id` 的请求 → 只取 `messages` 数组里**最后一条 user 消息**作为 prompt
- 第一次见到该 `session_id`：用 `--session-id <id>`（带 `--system-prompt`）创建
- 后续同一个 `session_id`：用 `--resume <id>`，**不再传** `--system-prompt`（session 已经有自己的 system）
- CLI 报"session 不存在"（比如服务重启、CLI 端被清）→ 自动 fallback 重建
- 同一 `session_id` 并发 → per-session `asyncio.Lock` 串行；不同 session 之间仍然并发

#### 用法约束

1. **同一 session 不能并发**：服务端会自动串行，但如果你期待并行，请用不同 session_id。
2. **不能回滚/编辑历史**：CLI 的 session 是 append-only。想要 regenerate 或分叉，就开新 session_id。
3. **二次调用的 system message 会被忽略**：第一次调用的 system 会绑死在 session 上，后续即使你在 messages 里改了 system，服务端也不会更新它（CLI 不支持中途换 system）。
4. **session_id 必须是合法 UUID**：CLI 的 `--session-id` 要求 UUID 格式，建议直接 `uuid.uuid4()`。

#### 不传 `session_id` 的客户端行为不变

老调用完全不受影响，依然走方式 1 的无状态路径。两种方式可以混用——比如长对话用 session，独立的批量任务用无状态。

---

## 并发 / 重试 / 超时调优

全部走环境变量，重启生效：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `CLAUDE_MAX_CONCURRENCY` | 4 | 同时跑几个 `claude` 子进程。账号容易触发限流就调小 |
| `CLAUDE_TIMEOUT` | 300 | 单次 CLI 调用超时秒数，超时强制 kill 子进程 |
| `CLAUDE_MAX_RETRIES` | 5 | 失败/超时重试次数 |
| `CLAUDE_BASE_BACKOFF` | 2.0 | 退避基数，第 n 次等 `2 * 2^(n-1)` 秒，封顶 60 |
| `CLAUDE_API_PORT` | 8765 | HTTP 端口 |
| `CLAUDE_DEFAULT_MODEL` | (无) | 客户端传占位词时的全局默认值 |
| `CLAUDE_BIN` | `claude` | CLI 可执行文件路径 |

```bash
CLAUDE_MAX_CONCURRENCY=2 CLAUDE_MAX_RETRIES=8 python claude_api.py
```

---

## 已知限制

- `temperature` / `top_p` / `max_tokens` 等参数在请求体里会被**忽略**（CLI 不暴露这些控制），但不会报错——纯做协议兼容。
- 流式是"伪流式"：服务端拿到完整结果后切片成 SSE chunks 推。这样 retry 能在生成阶段完整生效，不会出现"半句话已经发出去再失败"。
- 不做鉴权，建议只监听本机 `127.0.0.1`。要暴露到局域网请自己加反代 + token。
# CLI2API
