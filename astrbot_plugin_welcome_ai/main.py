"""AstrBot 欢迎与唤醒词 AI 插件。

功能：
1. 自动欢迎新成员：贴合人格生成欢迎语。
2. 自定义唤醒词：命中关键词后发送配置的回复文案，可用 AI 微调。
3. 频率感知：把关键词最近被触发的次数告诉 AI。
4. 主动发言：机器人自己沉默太久时主动说话。
5. 聊天模式：只响应唤醒词，或任何消息都回复。

AI 接入方式：
- direct_api（推荐）：插件自己请求 OpenAI 兼容的 HTTP API
- astrbot_provider：用 AstrBot 里配好的 provider
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star
from astrbot.core.message.message_event_result import MessageChain

try:
    from astrbot.api.message_components import At
    _AT_AVAILABLE = True
except ImportError:
    At = None  # type: ignore[assignment]
    _AT_AVAILABLE = False


PLUGIN_NAME = "astrbot_plugin_welcome_ai"

_DEFAULT_WELCOME_TEMPLATE = (
    "有新人「{member_name}」加入了群聊，请写一句简短、自然、"
    "符合你当前人格的欢迎语，不要复述要求。"
)
_FALLBACK_WELCOME = "欢迎 {member_name} 加入！"
_DEFAULT_IDLE_PROMPT = (
    "你已经很久没在这个群里说话了。现在主动说一句轻松的话，"
    "像老朋友一样随便聊一句。简短自然，符合你的性格。"
    "不要复述规则，只输出要说的话。"
)

_MODE_WAKE_ONLY = "wake_word_only"
_MODE_ALWAYS = "always_reply"
_BACKEND_DIRECT = "direct_api"
_BACKEND_PROVIDER = "astrbot_provider"

_MAX_HISTORY = 200


@dataclass(frozen=True, slots=True)
class WakeRule:
    keyword: str
    reply: str = ""
    prompt: str = ""


# ----------------------------------------------------------------------
# 配置解析
# ----------------------------------------------------------------------
def _parse_wake_words(raw: object) -> list[WakeRule]:
    if raw is None:
        return []
    if isinstance(raw, list):
        rules: list[WakeRule] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            keyword = str(item.get("keyword") or "").strip()
            reply = str(item.get("reply") or "").strip()
            prompt = str(item.get("prompt") or "").strip()
            if keyword:
                rules.append(
                    WakeRule(keyword=keyword, reply=reply, prompt=prompt)
                )
        return rules

    text = str(raw).strip()
    if not text:
        return []
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        parts = [piece.strip() for piece in line.split("|")]
        keyword = parts[0] if len(parts) >= 1 else ""
        reply = parts[1] if len(parts) >= 2 else ""
        prompt = parts[2] if len(parts) >= 3 else ""
        if keyword:
            rules.append(WakeRule(keyword=keyword, reply=reply, prompt=prompt))
    return rules


def _parse_whitelist(raw: object) -> set[str]:
    if raw is None:
        return set()
    text = (
        str(raw)
        .replace("，", ",").replace("；", ",").replace(";", ",")
        .replace("\r", "\n").replace("\n", ",")
    )
    result: set[str] = set()
    for piece in text.split(","):
        token = piece.strip()
        if token and token.isdigit():
            result.add(token)
    return result


# ----------------------------------------------------------------------
# 插件主体
# ----------------------------------------------------------------------
class WelcomeAIPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context, config)
        self._config = dict(config or {})
        self._wake_rules: list[WakeRule] = []
        self._whitelist: set[str] = set()
        self._last_wake: dict[str, float] = {}
        self._keyword_history: dict[tuple[str, str], list[float]] = {}
        # 主动发言用：
        # 记录机器人最后一次在某会话说话的时间
        self._last_bot_speech: dict[str, float] = {}
        # 记录机器人“见过”的会话（收到过消息的）
        self._known_sessions: set[str] = set()
        self._idle_task: asyncio.Task[None] | None = None
        self._http: aiohttp.ClientSession | None = None
        self._initialized = False
        self._refresh_config()

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        connector = aiohttp.TCPConnector(limit=4, ttl_dns_cache=300)
        self._http = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=int(self._config.get("api_timeout") or 30),
                connect=10,
            ),
        )

        backend = self._backend()
        if backend == _BACKEND_DIRECT:
            base = str(self._config.get("api_base_url") or "").strip()
            key = str(self._config.get("api_key") or "").strip()
            model = str(self._config.get("api_model") or "").strip()
            if not base or not key or not model:
                logger.warning(
                    "welcome-ai 直连 API 缺少配置：base_url=%s，key=%s，model=%s。",
                    "有" if base else "无",
                    "有" if key else "无",
                    "有" if model else "无",
                )
            else:
                logger.info(
                    "welcome-ai 已启用（直连 API）：base=%s，model=%s",
                    base, model,
                )
        else:
            self._log_provider_list()

        if bool(self._config.get("enable_idle_chat", False)):
            if self._idle_task is None or self._idle_task.done():
                self._idle_task = asyncio.create_task(
                    self._idle_loop(), name="welcome-ai-idle"
                )
                logger.info(
                    "welcome-ai 主动发言已启用：沉默 %s 秒后触发",
                    int(self._config.get("idle_seconds") or 1800),
                )

        logger.info(
            "welcome-ai 已启用：欢迎=%s，唤醒词=%d 条，AI 微调=%s，"
            "聊天模式=%s，白名单=%d 项，频率窗口=%s 秒",
            "开" if self._config.get("enable_welcome", True) else "关",
            len(self._wake_rules),
            "开" if self._config.get("enable_ai_polish", True) else "关",
            str(self._config.get("chat_mode") or _MODE_WAKE_ONLY),
            len(self._whitelist),
            int(self._config.get("freq_window_seconds") or 60),
        )

    async def terminate(self) -> None:
        self._initialized = False
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        self._idle_task = None

        if self._http is not None and not self._http.closed:
            try:
                await self._http.close()
            except Exception:
                pass
        self._http = None

    # ------------------------------------------------------------------
    # 基础配置
    # ------------------------------------------------------------------
    def _refresh_config(self) -> None:
        candidates: list[object] = []

        star_cfg = getattr(self, "config", None)
        if isinstance(star_cfg, dict):
            nested = star_cfg.get(PLUGIN_NAME)
            if isinstance(nested, dict):
                candidates.append(nested)
            candidates.append(star_cfg)

        try:
            live = self.context.get_config()
            if isinstance(live, dict):
                section = live.get(PLUGIN_NAME)
                if isinstance(section, dict):
                    candidates.append(section)
                nested = live.get("plugin")
                if isinstance(nested, dict):
                    section = nested.get(PLUGIN_NAME)
                    if isinstance(section, dict):
                        candidates.append(section)
        except Exception:
            pass

        for candidate in candidates:
            if isinstance(candidate, dict) and candidate:
                self._config.update(candidate)
                break

        self._wake_rules = _parse_wake_words(self._config.get("wake_words"))
        self._whitelist = _parse_whitelist(self._config.get("admin_whitelist"))

    def _backend(self) -> str:
        value = str(self._config.get("ai_backend") or _BACKEND_DIRECT).strip()
        if value not in (_BACKEND_DIRECT, _BACKEND_PROVIDER):
            return _BACKEND_DIRECT
        return value

    def _chat_mode(self) -> str:
        mode = str(self._config.get("chat_mode") or _MODE_WAKE_ONLY).strip()
        if mode not in (_MODE_WAKE_ONLY, _MODE_ALWAYS):
            return _MODE_WAKE_ONLY
        return mode

    def _system_prompt(self) -> str:
        return str(
            self._config.get("welcome_system_prompt")
            or "你是一个友善、活泼的群聊机器人，说话自然、简短。"
        ).strip()

    # ------------------------------------------------------------------
    # 频率统计
    # ------------------------------------------------------------------
    def _record_and_count(
        self, session: str, keyword: str, window_seconds: int
    ) -> int:
        now = time.monotonic()
        cutoff = now - window_seconds
        key = (session, keyword)

        history = self._keyword_history.setdefault(key, [])
        while history and history[0] < cutoff:
            history.pop(0)
        if len(history) >= _MAX_HISTORY:
            del history[: len(history) - _MAX_HISTORY + 1]
        history.append(now)
        return len(history)

    # ------------------------------------------------------------------
    # 机器人发言记录（用于主动发言判断）
    # ------------------------------------------------------------------
    def _mark_bot_spoke(self, session_id: str) -> None:
        """机器人每说一次话都调用它，刷新计时器。"""
        if not session_id:
            return
        self._last_bot_speech[session_id] = time.monotonic()
        self._known_sessions.add(session_id)

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_events(self, event: AstrMessageEvent):
        self._refresh_config()

        # 收到消息即认为机器人“见过”这个会话
        umo = event.unified_msg_origin or ""
        if umo:
            self._known_sessions.add(umo)
            # 首次见到这个会话时初始化计时器，避免立刻主动发言
            if umo not in self._last_bot_speech:
                self._last_bot_speech[umo] = time.monotonic()

        try:
            if await self._try_handle_group_increase(event):
                return
        except Exception:
            logger.exception("welcome-ai 处理群成员加入失败")

        try:
            await self._handle_message(event)
        except Exception:
            logger.exception("welcome-ai 处理消息失败")

    # ------------------------------------------------------------------
    # 群成员加入
    # ------------------------------------------------------------------
    async def _try_handle_group_increase(self, event: AstrMessageEvent) -> bool:
        if not self._config.get("enable_welcome", True):
            return False

        raw = getattr(event, "message_obj", None)
        raw_message = getattr(raw, "raw_message", None) if raw else None
        if not isinstance(raw_message, dict):
            return False
        if raw_message.get("post_type") != "notice":
            return False
        if raw_message.get("notice_type") != "group_increase":
            return False

        user_id = str(raw_message.get("user_id") or "").strip()
        member_name = await self._fetch_member_name(event, user_id)
        text = await self._make_welcome_text(event, member_name)
        if not text:
            return False
        await event.send(event.plain_result(text))
        self._mark_bot_spoke(event.unified_msg_origin or "")
        event.stop_event()
        return True

    async def _fetch_member_name(self, event: AstrMessageEvent, user_id: str) -> str:
        if not user_id:
            return "新成员"
        try:
            api = getattr(getattr(event, "bot", None), "api", None)
            group_id = event.get_group_id()
            if api is not None and group_id:
                info = await api.call_action(
                    "get_group_member_info",
                    group_id=int(group_id),
                    user_id=int(user_id),
                    no_cache=False,
                )
                if isinstance(info, dict):
                    name = str(
                        info.get("card") or info.get("nickname") or ""
                    ).strip()
                    if name:
                        return name
        except Exception:
            pass
        return f"用户{user_id}"

    async def _make_welcome_text(
        self, event: AstrMessageEvent, member_name: str
    ) -> str:
        template = (
            str(self._config.get("welcome_prompt_template") or "").strip()
            or _DEFAULT_WELCOME_TEMPLATE
        )
        prompt = template.replace("{member_name}", member_name)
        extra = str(self._config.get("welcome_extra") or "").strip()
        if extra:
            prompt = f"{prompt}\n{extra}"

        if self._config.get("enable_ai_polish", True):
            text = await self._ask_ai(event.unified_msg_origin or "", prompt)
            if text:
                return text
        return _FALLBACK_WELCOME.format(member_name=member_name)

    # ------------------------------------------------------------------
    # 消息处理
    # ------------------------------------------------------------------
    async def _handle_message(self, event: AstrMessageEvent) -> None:
        message = (event.message_str or "").strip()
        rule = self._match_wake_rule(message) if message else None

        if rule is not None:
            if self._config.get("wake_only_group", False) and not event.get_group_id():
                self._apply_chat_mode(event)
                return
            if not self._pass_cooldown(event):
                return
            text = await self._make_wake_reply(event, rule)
            if text:
                await event.send(event.plain_result(text))
                self._mark_bot_spoke(event.unified_msg_origin or "")
                event.should_call_llm(False)
                event.stop_event()
                return

        self._apply_chat_mode(event)

    def _apply_chat_mode(self, event: AstrMessageEvent) -> None:
        if self._chat_mode() == _MODE_ALWAYS:
            event.should_call_llm(True)
            # 交给 AstrBot 默认流程回复，我们无法直接知道它何时说完，
            # 但可以在下一次收到消息时更新计时器（见 on_all_events）

    def _match_wake_rule(self, message: str) -> WakeRule | None:
        for rule in sorted(
            self._wake_rules, key=lambda r: len(r.keyword), reverse=True
        ):
            if rule.keyword in message:
                return rule
        return None

    def _pass_cooldown(self, event: AstrMessageEvent) -> bool:
        seconds = int(self._config.get("cooldown_seconds") or 0)
        if seconds <= 0:
            return True
        key = event.unified_msg_origin or "default"
        now = time.monotonic()
        last = self._last_wake.get(key, 0.0)
        if now - last < seconds:
            return False
        self._last_wake[key] = now
        return True

    async def _make_wake_reply(
        self, event: AstrMessageEvent, rule: WakeRule
    ) -> str:
        has_reply = bool(rule.reply)
        has_prompt = bool(rule.prompt)

        if not has_reply and not has_prompt:
            if self._chat_mode() == _MODE_ALWAYS:
                event.should_call_llm(True)
            return ""

        if not self._config.get("wake_use_ai", True):
            return rule.reply

        session = event.unified_msg_origin or "default"
        window = int(self._config.get("freq_window_seconds") or 60)
        count = self._record_and_count(session, rule.keyword, window)

        ai_prompt = self._build_wake_prompt(
            rule, count=count, window=window
        )
        text = await self._ask_ai(session, ai_prompt)
        if text:
            return text
        return rule.reply

    @staticmethod
    def _build_wake_prompt(
        rule: WakeRule, *, count: int, window: int
    ) -> str:
        has_reply = bool(rule.reply)
        has_prompt = bool(rule.prompt)

        freq_hint = (
            f"【频率信息】这是关键词「{rule.keyword}」在最近 {window} 秒内"
            f"第 {count} 次被触发。\n"
        )

        if has_reply and has_prompt:
            return (
                freq_hint
                + "\n请按下面的要求改写一段回复文案，保持原意，"
                "不要加解释，只输出改写后的内容。\n\n"
                f"【原始文案】\n{rule.reply}\n\n"
                f"【额外要求】\n{rule.prompt}"
            )
        if has_reply:
            return (
                freq_hint
                + "\n请按你的系统提示词风格改写下面这段回复文案，"
                "保持原意，不要加解释，只输出改写后的内容：\n\n"
                f"{rule.reply}"
            )
        return (
            freq_hint
            + "\n请按下面的要求生成一条简短、自然的回复，"
            "不要加解释，只输出回复内容：\n\n"
            f"{rule.prompt}"
        )

    # ------------------------------------------------------------------
    # 主动发言
    # ------------------------------------------------------------------
    async def _idle_loop(self) -> None:
        """后台循环：定期检查哪些会话机器人已沉默太久。"""
        try:
            while True:
                interval = max(
                    10, int(self._config.get("idle_check_interval") or 60)
                )
                try:
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    raise

                try:
                    await self._idle_tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("welcome-ai 主动发言检查失败")
        except asyncio.CancelledError:
            raise

    async def _idle_tick(self) -> None:
        if not bool(self._config.get("enable_idle_chat", False)):
            return

        idle_seconds = max(60, int(self._config.get("idle_seconds") or 1800))
        cooldown = max(
            60, int(self._config.get("idle_cooldown_seconds") or 3600)
        )
        only_group = bool(self._config.get("idle_only_group", True))
        now = time.monotonic()

        for session_id in list(self._known_sessions):
            if only_group and "GroupMessage" not in session_id:
                continue
            # 该会话机器人最后一次发言距今多久
            last_speech = self._last_bot_speech.get(session_id)
            if last_speech is None:
                continue
            if now - last_speech < idle_seconds:
                continue
            # 检查冷却：防止连续主动发言
            # 这里复用 _last_bot_speech：因为主动发言后我们会更新它
            try:
                await self._do_idle_chat(session_id)
            except Exception:
                logger.exception(
                    "welcome-ai 主动发言失败：%s", session_id
                )

    async def _do_idle_chat(self, session_id: str) -> None:
        prompt = (
            str(self._config.get("idle_prompt") or "").strip()
            or _DEFAULT_IDLE_PROMPT
        )
        text = await self._ask_ai(session_id, prompt)
        if not text:
            logger.info("welcome-ai 主动发言跳过（AI 无输出）：%s", session_id)
            # 即便 AI 没输出也刷新一次计时器，避免每轮都重试
            self._mark_bot_spoke(session_id)
            return

        try:
            await self.context.send_message(
                session_id, MessageChain([Plain(text)])
            )
            self._mark_bot_spoke(session_id)
            logger.info(
                "welcome-ai 主动发言：%s → %s", session_id, text[:40]
            )
        except Exception:
            logger.exception("welcome-ai 主动发言发送失败：%s", session_id)

    # ------------------------------------------------------------------
    # 命令：欢迎 @某人
    # ------------------------------------------------------------------
    @filter.command("欢迎")
    async def cmd_welcome(self, event: AstrMessageEvent, arg: str = ""):
        if not self._is_admin(event):
            yield event.plain_result(
                "该命令仅限管理员使用。\n"
                "请在插件配置的“管理员白名单”里添加你的用户号或群号。"
            )
            event.stop_event()
            return

        member_name = ""
        if _AT_AVAILABLE:
            for comp in getattr(event.message_obj, "message", None) or []:
                if isinstance(comp, At):
                    qq = getattr(comp, "qq", None) or getattr(comp, "target", None)
                    if qq is not None:
                        member_name = await self._fetch_member_name(
                            event, str(qq)
                        )
                        break
        if not member_name:
            member_name = (arg or "").strip()
        if not member_name:
            yield event.plain_result("用法：欢迎 @某人  或  欢迎 昵称")
            event.stop_event()
            return

        text = await self._make_welcome_text(event, member_name)
        yield event.plain_result(text)
        event.stop_event()

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        if not self._whitelist:
            return False
        user_id = _event_str(event, "get_sender_id")
        group_id = _event_str(event, "get_group_id")
        if user_id and user_id in self._whitelist:
            return True
        if group_id and group_id in self._whitelist:
            return True
        return False

    # ------------------------------------------------------------------
    # AI：统一入口
    # ------------------------------------------------------------------
    async def _ask_ai(self, session_id: str, prompt: str) -> str:
        backend = self._backend()
        if backend == _BACKEND_PROVIDER:
            return await self._ask_via_provider(session_id, prompt)
        return await self._ask_via_direct_api(prompt)

    async def _ask_via_direct_api(self, prompt: str) -> str:
        if self._http is None or self._http.closed:
            return ""

        base = str(self._config.get("api_base_url") or "").strip().rstrip("/")
        key = str(self._config.get("api_key") or "").strip()
        model = str(self._config.get("api_model") or "").strip()
        if not base or not key or not model:
            logger.warning("welcome-ai 直连 API 配置不完整，跳过 AI 调用")
            return ""

        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        try:
            temperature = float(self._config.get("api_temperature") or 0.8)
        except (TypeError, ValueError):
            temperature = 0.8

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        timeout_seconds = int(self._config.get("api_timeout") or 30)

        try:
            async with self._http.post(
                url,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds, connect=10),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning(
                        "welcome-ai API 返回 HTTP %s：%s",
                        resp.status, text[:200],
                    )
                    return ""
                try:
                    payload = json.loads(text)
                except Exception:
                    logger.warning("welcome-ai API 返回非 JSON：%s", text[:200])
                    return ""
        except aiohttp.ClientError as exc:
            logger.warning("welcome-ai API 网络错误：%s", exc)
            return ""
        except Exception:
            logger.exception("welcome-ai API 调用失败")
            return ""

        return _extract_content(payload)

    def _get_provider(self, session_id: str):
        preferred = str(self._config.get("preferred_provider_id") or "").strip()
        if preferred:
            try:
                provider = self.context.get_provider_by_id(pid=preferred)
                if provider is not None:
                    return provider
            except Exception:
                pass

        try:
            pid = self.context.get_current_chat_provider_id(umo=session_id)
            if pid:
                provider = self.context.get_provider_by_id(pid=pid)
                if provider is not None:
                    return provider
        except Exception:
            pass

        for call in (
            lambda: self.context.get_using_provider(umo=session_id),
            lambda: self.context.get_using_provider(),
        ):
            try:
                provider = call()
                if provider is not None:
                    return provider
            except Exception:
                continue

        try:
            providers = self.context.get_all_providers()
            if providers:
                return providers[0]
        except Exception:
            pass
        return None

    async def _ask_via_provider(self, session_id: str, prompt: str) -> str:
        provider = self._get_provider(session_id)
        if provider is None:
            logger.warning("welcome-ai 找不到 AstrBot provider，跳过 AI 调用")
            return ""

        chat_session_id = f"welcome-ai-{session_id}"
        resp = None
        try:
            resp = await provider.text_chat(
                prompt=prompt,
                session_id=chat_session_id,
                persist=False,
                system_prompt=self._system_prompt(),
            )
        except TypeError:
            try:
                resp = await provider.text_chat(
                    prompt=prompt, session_id=chat_session_id
                )
            except Exception:
                logger.exception("welcome-ai provider.text_chat 失败")
                return ""
        except Exception:
            logger.exception("welcome-ai provider.text_chat 失败")
            return ""

        text = getattr(resp, "completion_text", "") or ""
        return str(text).strip()

    def _log_provider_list(self) -> None:
        try:
            providers = self.context.get_all_providers()
        except Exception:
            return
        if not providers:
            logger.warning("welcome-ai 没有检测到任何 AstrBot provider")
            return
        logger.info("welcome-ai 可用 provider 列表：")
        for provider in providers:
            meta = getattr(provider, "meta", None)
            pid = str(getattr(meta, "id", "") or "?").strip()
            model = str(getattr(meta, "model", "") or "?").strip()
            logger.info("  - id=%s  model=%s", pid, model)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _extract_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for piece in content:
                if isinstance(piece, dict):
                    text = piece.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts).strip()
    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _event_str(event: AstrMessageEvent, method_name: str) -> str | None:
    method = getattr(event, method_name, None)
    if not callable(method):
        return None
    try:
        value = method()
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None