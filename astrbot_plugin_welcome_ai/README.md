# 欢迎与唤醒词AI

AstrBot 插件：自动生成贴合人格的欢迎新成员文案；支持自定义唤醒词，回复由 AI 微调。

## 功能

- **自动欢迎新成员**：群成员加入时，AI 参考系统提示词生成欢迎语
- **自定义唤醒词**：命中关键词后，可让 AI 按你的要求生成/改写回复
- **AI 要求**：每条唤醒词可以额外写"AI 要遵守什么"
- **聊天模式**：可切换"只响应唤醒词"或"任何消息都回复"
- **直连 API**：不依赖 AstrBot provider
- **管理员手动欢迎**：白名单用户可用 `欢迎 @某人`

## 安装

1. 把本插件目录放到 `AstrBot/data/plugins/astrbot_plugin_welcome_ai/`
2. 安装依赖：`python -m pip install aiohttp`
3. 重启 AstrBot

## 配置（直连 API）

最少填三个字段：

| 配置项 | 填什么 |
| --- | --- |
| **API 地址** | `https://api.deepseek.com/v1`（或其他兼容厂商） |
| **API 密钥** | 你的密钥（以 `sk-` 开头） |
| **模型名称** | `deepseek-chat`（或其他模型名） |

### 各厂商示例

| 厂商 | API 地址 | 模型名 |
| --- | --- | --- |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` 或 `deepseek-flash` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |

### 系统提示词（人格设定）

配置里的 **AI 系统提示词** 影响 AI 生成欢迎语和微调回复时的整体风格。
