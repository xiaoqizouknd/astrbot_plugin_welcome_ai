"""AstrBot 欢迎与唤醒词 AI 插件。

功能：
1. 自动欢迎新成员：短时间连续入群自动合并成一条欢迎语，贴合人格。
2. 自定义唤醒词：命中关键词后发送配置的回复文案，可用 AI 微调；
   提示词会带上说话人和原话，AI 回复更贴合场景。
3. 回复概率：每条唤醒词可设置 0-100 的回复概率，避免刷屏。
4. 频率感知与上限：命中即记录触发次数并告诉 AI；
   触发超过上限后插件自动闭嘴。
5. 主动发言：机器人自己沉默太久时主动说话，检查间隔带随机抖动。
6. 聊天模式：只响应唤醒词，或任何消息都回复。
7. 输出长度限制：所有 AI 输出都受 max_reply_length 控制。
8. 防风控：随机回复延迟、全局/会话发送间隔、每分钟回复上限、
   AI 文案去重、夜间静默、忽略机器人自身消息，更像真人、更稳。
9. AI 直连 API 失败自动重试（网络错误 / 429 / 5xx，指数退避）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

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

# 与 _conf_schema.json 里的 default 保持一致
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
_DEFAULT_SYSTEM_PROMPT = (
    "你是一个友善、活泼的群聊机器人，说话自然、简短，喜欢用轻松的语气。"
)

_MODE_WAKE_ONLY = "wake_word_only"
_MODE_ALWAYS = "always_reply"
_BACKEND_DIRECT = "direct_api"
_BACKEND_PROVIDER = "astrbot_provider"

# 每个 (会话, 关键词) 保留的触发历史条数上限
_MAX_HISTORY = 200
# 每个会话记住最近几条已发送文案，用于去重
_MAX_RECENT_TEXTS = 6
# 会话状态清理阈值：超过这个时间无活动就丢弃记录
_STALE_SESSION_SECONDS = 86400.0
# 每处理多少条事件触发一次清理（独立于主动发言开关）
_CLEANUP_EVERY_N_EVENTS = 100
# 直连 API 网络 / 5xx 失败时的额外重试次数
_API_RETRIES = 2
# 拼进唤醒词提示词的原话长度上限
_PROMPT_CONTEXT_LIMIT = 200
# 合并欢迎一次最多带的人数
_WELCOME_NAME_LIMIT = 20


@dataclass(frozen=True, slots=True)
class WakeRule:
    keyword: str
    reply: str = ""
    prompt: str = ""
    probability: int = 100


# ----------------------------------------------------------------------
# 配置解析
# ----------------------------------------------------------------------
def _parse_probability(value: object) -> int:
    if value is None:
        return 100
    text = str(value).strip()
    if not text:
        return 100
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        f = float(text)
    except (TypeError, ValueError):
        return 100
    # 只有“带小数点且小于 1”的写法才按 0~1 比例理解（0.8 → 80%）。
    # 整数一律按百分比理解（1 → 1%），避免 “1” 被误当成 100%。
    if "." in text and 0 < f < 1:
        f = f * 100
    return max(0, min(100, int(round(f))))


def _parse_wake_words(raw: object) -> list[WakeRule]:
    if raw is None:
        return []

    # 单条规则以 dict 形态出现时也兼容
    if isinstance(raw, dict):
        keyword = str(raw.get("keyword") or "").strip()
        if not keyword:
            return []
        return [
            WakeRule(
                keyword=keyword,
                reply=str(raw.get("reply") or "").strip(),
                prompt=str(raw.get("prompt") or "").strip(),
                probability=_parse_probability(raw.get("probability")),
            )
        ]

    if isinstance(raw, list):
        rules: list[WakeRule] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            keyword = str(item.get("keyword") or "").strip()
            reply = str(item.get("reply") or "").strip()
            prompt = str(item.get("prompt") or "").strip()
            probability = _parse_probability(item.get("probability"))
            if keyword:
                rules.append(
                    WakeRule(
                        keyword=keyword,
                        reply=reply,
                        prompt=prompt,
                        probability=probability,
                    )
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
        probability = _parse_probability(parts[3]) if len(parts) >= 4 else 100
        if keyword:
            rules.append(
                WakeRule(
                    keyword=keyword,
                    reply=reply,
                    prompt=prompt,
                    probability=probability,
                )
            )
    return rules


def _parse_whitelist(raw: object) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(v).strip() for v in raw if str(v).strip().isdigit()}
    if isinstance(raw, dict):
        return {str(v).strip() for v in raw.values() if str(v).strip().isdigit()}
    text = (
        str(raw)
        .replace("，", ",").replace("；", ",").replace(";", ",")
        .replace("\r", "\n").replace("\n", ",")
    )
    return {t.strip() for t in text.split(",") if t.strip().isdigit()}


def _parse_hhmm(text: str) -> tuple[int, int]:
    h, m = text.split(":", 1)
    h = int(h)
    m = int(m)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError
    return h, m


def _parse_quiet_hours(text: str) -> tuple[int, int] | None:
    """解析 “23:30-07:00” 形式的静默时段，返回 (开始, 结束) 的分钟数。

    跨天时段合法（如 23:30-07:00）。留空 / off 表示不静默。
    """
    text = (text or "").strip()
    if not text or text.lower() in ("off", "none", "关闭"):
        return None
    parts = [p.strip() for p in text.replace("：", ":").split("-", 1)]
    if len(parts) != 2:
        return None
    try:
        h1, m1 = _parse_hhmm(parts[0])
        h2, m2 = _parse_hhmm(parts[1])
    except (TypeError, ValueError):
        return None
    return h1 * 60 + m1, h2 * 60 + m2


def _to_int(value: object, default: int, *, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, n)


def _to_float(value: object, default: float, *, minimum: float = 0.0) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, f)


# ----------------------------------------------------------------------
# 插件主体
# ----------------------------------------------------------------------
class WelcomeAIPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context, config)
        self._config: dict[str, Any] = dict(config or {})

        self._http: aiohttp.ClientSession | None = None
        self._initialized = False
        # 探测结果缓存：是否支持 get_self_id（None 表示未探测）
        self._supports_self_id: bool | None = None
        self._idle_task: asyncio.Task[None] | None = None
        # 合并欢迎：会话 -> 待欢迎名单 / 发送任务
        self._pending_welcome: dict[str, list[str]] = {}
        self._welcome_tasks: dict[str, asyncio.Task[None]] = {}
        # 清理计数器：每 N 条事件触发一次 _cleanup_stale
        self._cleanup_counter = 0

        # ---- 会话级状态（统一由 _cleanup_stale 清理，避免无界增长）----
        self._known_sessions: set[str] = set()
        self._last_bot_speech: dict[str, float] = {}
        self._last_idle_chat: dict[str, float] = {}
        self._last_wake: dict[str, float] = {}
        self._keyword_history: dict[tuple[str, str], deque[float]] = {}
        # 防风控状态
        self._recent_texts: dict[str, deque[str]] = {}
        self._send_budget: dict[str, deque[float]] = {}
        self._last_send_session: dict[str, float] = {}
        self._last_send_global = 0.0

        self._apply_config()

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def _refresh_config(self) -> None:
        """合并最新配置并重建派生数据。

        优先级：``self.config``（AstrBot 保存配置时会热更新它）
        > ``context.get_config()`` 里的 plugin 段。
        只有 ``self.config`` 为空时才去查 context，
        避免每条消息都做一次全量配置读取。
        """
        merged: dict[str, Any] = {}

        try:
            star_cfg = getattr(self, "config", None)
        except Exception:
            star_cfg = None
        if isinstance(star_cfg, dict):
            merged.update(star_cfg)
            nested = star_cfg.get(PLUGIN_NAME)
            if isinstance(nested, dict):
                merged.update(nested)

        if not merged:
            try:
                live = self.context.get_config()
                if isinstance(live, dict):
                    section = live.get(PLUGIN_NAME)
                    if isinstance(section, dict):
                        merged.update(section)
                    plugin_section = live.get("plugin")
                    if isinstance(plugin_section, dict):
                        section = plugin_section.get(PLUGIN_NAME)
                        if isinstance(section, dict):
                            merged.update(section)
            except Exception:
                pass

        if merged:
            self._config.update(merged)

        self._apply_config()

    def _apply_config(self) -> None:
        """把原始配置解析成类型化字段，之后只查缓存值。"""
        cfg = self._config

        self.backend = str(cfg.get("ai_backend") or _BACKEND_DIRECT).strip()
        if self.backend not in (_BACKEND_DIRECT, _BACKEND_PROVIDER):
            self.backend = _BACKEND_DIRECT

        self.chat_mode = str(cfg.get("chat_mode") or _MODE_WAKE_ONLY).strip()
        if self.chat_mode not in (_MODE_WAKE_ONLY, _MODE_ALWAYS):
            self.chat_mode = _MODE_WAKE_ONLY

        self.system_prompt = (
            str(cfg.get("welcome_system_prompt") or "").strip()
            or _DEFAULT_SYSTEM_PROMPT
        )
        self.max_reply_length = _to_int(cfg.get("max_reply_length"), 0)
        self.api_timeout = _to_int(cfg.get("api_timeout"), 30, minimum=1)
        self.freq_window = _to_int(cfg.get("freq_window_seconds"), 60, minimum=1)
        self.cooldown_seconds = _to_int(cfg.get("cooldown_seconds"), 0)
        self.idle_seconds = _to_int(cfg.get("idle_seconds"), 1800, minimum=60)
        self.idle_check_interval = _to_int(cfg.get("idle_check_interval"), 60, minimum=10)
        self.idle_cooldown_seconds = _to_int(cfg.get("idle_cooldown_seconds"), 3600)

        # 防风控参数
        self.welcome_merge_seconds = _to_int(cfg.get("welcome_merge_seconds"), 3)
        self.max_triggers_per_window = _to_int(cfg.get("max_triggers_per_window"), 0)
        self.reply_delay_min = _to_float(cfg.get("reply_delay_min"), 0.5)
        self.reply_delay_max = _to_float(cfg.get("reply_delay_max"), 1.5)
        if self.reply_delay_max < self.reply_delay_min:
            self.reply_delay_max = self.reply_delay_min
        self.send_min_interval = _to_float(cfg.get("send_min_interval"), 0.5)
        self.max_replies_per_minute = _to_int(cfg.get("max_replies_per_minute"), 10)

        # 唤醒词：解析后按长度倒序建索引，匹配时只遍历不排序
        rules = _parse_wake_words(cfg.get("wake_words"))
        self.wake_index: list[WakeRule] = sorted(
            rules, key=lambda r: len(r.keyword), reverse=True
        )
        self.whitelist = _parse_whitelist(cfg.get("admin_whitelist"))
        self.quiet_range = _parse_quiet_hours(str(cfg.get("quiet_hours") or ""))

    def _quiet_desc(self) -> str:
        if self.quiet_range is None:
            return "关"
        start, end = self.quiet_range
        return (
            f"{start // 60:02d}:{start % 60:02d}-"
            f"{end // 60:02d}:{end % 60:02d}"
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        connector = aiohttp.TCPConnector(limit=4, ttl_dns_cache=300)
        self._http = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=self.api_timeout,
                connect=10,
            ),
        )

        # 探测 AstrBot 版本是否支持 get_self_id（用于识别并忽略机器人自身消息）
        self._supports_self_id = hasattr(AstrMessageEvent, "get_self_id")
        if self._supports_self_id:
            logger.info(
                "welcome-ai 支持 get_self_id，可自动识别并忽略机器人自身消息"
            )
        else:
            logger.info(
                "welcome-ai 不支持 get_self_id，主动发言只跟踪插件自身发言"
                "（核心 LLM 回复不会被计入沉默计时）"
            )

        if self.backend == _BACKEND_DIRECT:
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

        # 主动发言任务按当前配置启停；运行中改配置也会即时生效
        await self._sync_idle_task()

        probs = [r.probability for r in self.wake_index]
        if probs and any(p < 100 for p in probs):
            logger.info(
                "welcome-ai 唤醒词回复概率：%d 条自定义，其余按 100%%",
                sum(1 for p in probs if p < 100),
            )

        if self.max_reply_length > 0:
            logger.info("welcome-ai 输出长度限制：%d 字符", self.max_reply_length)
        else:
            logger.info("welcome-ai 输出长度限制：不限制")

        logger.info(
            "welcome-ai 已启用：欢迎=%s，唤醒词=%d 条，AI 微调=%s，"
            "聊天模式=%s，白名单=%d 项，频率窗口=%s 秒，静默=%s",
            "开" if self._config.get("enable_welcome", True) else "关",
            len(self.wake_index),
            "开" if self._config.get("enable_ai_polish", True) else "关",
            self.chat_mode,
            len(self.whitelist),
            self.freq_window,
            self._quiet_desc(),
        )

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        """取消并等待任务结束；已结束的任务取回异常，避免告警泄漏。"""
        if task is None:
            return
        if task.done():
            if not task.cancelled():
                try:
                    task.exception()
                except Exception:
                    pass
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def terminate(self) -> None:
        self._initialized = False

        await self._cancel_task(self._idle_task)
        self._idle_task = None

        for task in list(self._welcome_tasks.values()):
            await self._cancel_task(task)
        self._welcome_tasks.clear()
        self._pending_welcome.clear()

        if self._http is not None and not self._http.closed:
            try:
                await self._http.close()
            except Exception:
                pass
        self._http = None

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_events(self, event: AstrMessageEvent):
        self._refresh_config()

        # 主动发言开关热更新：配置变化时同步启停后台任务
        try:
            await self._sync_idle_task()
        except Exception:
            logger.exception("welcome-ai 同步主动发言任务失败")

        self._maybe_cleanup()
        self._track_session(event)

        try:
            if await self._try_handle_group_increase(event):
                return
        except Exception:
            logger.exception("welcome-ai 处理群成员加入失败")

        try:
            await self._handle_message(event)
        except Exception:
            logger.exception("welcome-ai 处理消息失败")

    def _maybe_cleanup(self) -> None:
        # 挂在这里而不是 _idle_tick，是为了在 enable_idle_chat=False 时也能清理
        self._cleanup_counter += 1
        if self._cleanup_counter >= _CLEANUP_EVERY_N_EVENTS:
            self._cleanup_counter = 0
            try:
                self._cleanup_stale(time.monotonic())
            except Exception:
                logger.exception("welcome-ai 清理会话状态失败")

    def _track_session(self, event: AstrMessageEvent) -> None:
        umo = event.unified_msg_origin or ""
        if not umo:
            return
        self._known_sessions.add(umo)
        if umo not in self._last_bot_speech:
            self._last_bot_speech[umo] = time.monotonic()
        # 发送者就是机器人时重置沉默计时（核心 LLM / 其他插件发出的也算）
        if self._is_self_message(event):
            self._mark_bot_spoke(umo)

    def _is_self_message(self, event: AstrMessageEvent) -> bool:
        if not self._supports_self_id:
            return False
        sender_id = _event_str(event, "get_sender_id")
        self_id = _event_str(event, "get_self_id")
        return bool(sender_id and self_id and sender_id == self_id)

    # ------------------------------------------------------------------
    # 静默时段
    # ------------------------------------------------------------------
    def _in_quiet_time(self, now_mins: int | None = None) -> bool:
        if self.quiet_range is None:
            return False
        start, end = self.quiet_range
        if now_mins is None:
            lt = time.localtime()
            now_mins = lt.tm_hour * 60 + lt.tm_min
        if start < end:
            return start <= now_mins < end
        # 跨天（如 23:30-07:00）；start == end 视为全天静默
        return now_mins >= start or now_mins < end

    # ------------------------------------------------------------------
    # 防风控：节奏 / 上限 / 去重 / 统一发送入口
    # ------------------------------------------------------------------
    def _consume_send_budget(self, session: str) -> bool:
        """检查“每个会话每分钟回复上限”，放行则消耗一次额度。"""
        cap = self.max_replies_per_minute
        if cap <= 0 or not session:
            return True
        dq = self._send_budget.setdefault(session, deque())
        now = time.monotonic()
        cutoff = now - 60.0
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= cap:
            logger.debug(
                "welcome-ai 每分钟回复上限（%d）已到：%s", cap, session
            )
            return False
        dq.append(now)
        return True

    async def _pace(self, session: str) -> None:
        """发送前等待：随机延迟 + 全局/会话最小发送间隔，模拟人类节奏。"""
        now = time.monotonic()
        delay = 0.0
        if self.reply_delay_max > 0:
            lo = min(self.reply_delay_min, self.reply_delay_max)
            hi = max(self.reply_delay_min, self.reply_delay_max)
            delay = random.uniform(lo, hi)
        gap = 0.0
        if self.send_min_interval > 0:
            gap = max(gap, self.send_min_interval - (now - self._last_send_global))
            if session:
                since = now - self._last_send_session.get(session, 0.0)
                gap = max(gap, self.send_min_interval - since)
        wait = max(0.0, delay, gap)
        if wait > 0:
            await asyncio.sleep(wait)
        t = time.monotonic()
        self._last_send_global = t
        if session:
            self._last_send_session[session] = t

    def _is_recent_dup(self, session: str, text: str) -> bool:
        dq = self._recent_texts.get(session)
        return bool(dq and text in dq)

    def _mark_recent(self, session: str, text: str) -> None:
        if session:
            self._recent_texts.setdefault(
                session, deque(maxlen=_MAX_RECENT_TEXTS)
            ).append(text)

    async def _send_managed(
        self,
        session: str,
        text: str,
        sender: Callable[[str], Awaitable[object]],
        *,
        dedupe: bool,
    ) -> bool:
        """统一发送入口：去重 → 速率上限 → 随机节奏 → 发送。

        dedupe=True 用于 AI 生成的文案；固定文案由用户自行控制是否重复。
        """
        text = (text or "").strip()
        if not text:
            return False
        if dedupe and self._is_recent_dup(session, text):
            logger.debug(
                "welcome-ai 跳过重复文案（%s）：%s", session, text[:30]
            )
            return False
        if not self._consume_send_budget(session):
            return False
        await self._pace(session)
        try:
            await sender(text)
        except Exception:
            logger.exception("welcome-ai 发送失败：%s", session)
            return False
        if dedupe:
            self._mark_recent(session, text)
        return True

    # ------------------------------------------------------------------
    # 群成员加入（合并欢迎）
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
        umo = event.unified_msg_origin or ""

        if self._in_quiet_time():
            logger.info("welcome-ai 静默时段，跳过欢迎：%s", member_name)
            return False

        if not umo:
            # 拿不到会话标识时直接欢迎，不合并
            text = await self._make_welcome_text(umo, member_name)
            if text:
                ok = await self._send_managed(
                    umo, text,
                    lambda t: event.send(event.plain_result(t)),
                    dedupe=True,
                )
                if ok:
                    self._mark_bot_spoke(umo)
            event.stop_event()
            return True

        names = self._pending_welcome.setdefault(umo, [])
        if len(names) < _WELCOME_NAME_LIMIT:
            names.append(member_name)
        event.stop_event()

        task = self._welcome_tasks.get(umo)
        if task is None or task.done():
            self._welcome_tasks[umo] = asyncio.create_task(
                self._flush_welcome(umo), name=f"welcome-ai-{umo}"
            )
        return True

    async def _flush_welcome(self, umo: str) -> None:
        try:
            if self.welcome_merge_seconds > 0:
                await asyncio.sleep(self.welcome_merge_seconds)
            names = self._pending_welcome.pop(umo, [])
            self._welcome_tasks.pop(umo, None)
            if not names or self._in_quiet_time():
                return
            joined = "、".join(dict.fromkeys(names))
            text = await self._make_welcome_text(umo, joined)
            if not text:
                return
            ok = await self._send_managed(
                umo, text,
                lambda t: self.context.send_message(
                    umo, MessageChain([Plain(t)])
                ),
                dedupe=True,
            )
            if ok:
                self._mark_bot_spoke(umo)
                logger.info("welcome-ai 欢迎：%s → %s", umo, joined)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("welcome-ai 合并欢迎发送失败：%s", umo)

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

    async def _make_welcome_text(self, session_id: str, member_name: str) -> str:
        template = (
            str(self._config.get("welcome_prompt_template") or "").strip()
            or _DEFAULT_WELCOME_TEMPLATE
        )
        prompt = template.replace("{member_name}", member_name)
        extra = str(self._config.get("welcome_extra") or "").strip()
        if extra:
            prompt = f"{prompt}\n{extra}"

        if self._config.get("enable_ai_polish", True):
            text = await self._ask_ai(session_id, prompt)
            if text:
                return text
        return _FALLBACK_WELCOME.format(member_name=member_name)

    # ------------------------------------------------------------------
    # 消息处理
    # ------------------------------------------------------------------
    async def _handle_message(self, event: AstrMessageEvent) -> None:
        # 不处理机器人自己的消息，避免自触发循环（如欢迎语里恰好含唤醒词）
        if self._is_self_message(event):
            return
        message = (event.message_str or "").strip()
        if not message:
            return

        rule = self._match_wake_rule(message)
        if rule is None:
            self._apply_chat_mode(event)
            return

        # 频率统计：命中即记录，包括之后被概率/冷却跳过的触发，
        # 让 AI 感知关键词的真实热度。
        session = event.unified_msg_origin or "default"
        count = self._record_and_count(session, rule.keyword, self.freq_window)

        # 静默时段：完全闭嘴（也不把消息交给核心 LLM）
        if self._in_quiet_time():
            return

        if self._config.get("wake_only_group", False) and not event.get_group_id():
            self._apply_chat_mode(event)
            return

        # 触发次数超过上限：插件闭嘴防刷屏
        if 0 < self.max_triggers_per_window < count:
            logger.debug(
                "welcome-ai 唤醒词「%s」%d 秒内触发 %d 次，超过上限 %d，跳过",
                rule.keyword, self.freq_window, count, self.max_triggers_per_window,
            )
            self._apply_chat_mode(event)
            return

        if not self._pass_probability(rule):
            logger.debug(
                "welcome-ai 唤醒词「%s」概率跳过（%d%%）",
                rule.keyword, rule.probability,
            )
            self._apply_chat_mode(event)
            return

        # 冷却未过：让 always 模式仍能把事件交给核心 LLM，
        # 避免"命中唤醒词但没回复"的空白。
        if not self._check_cooldown(event):
            self._apply_chat_mode(event)
            return

        text = await self._make_wake_reply(event, rule, count=count, message=message)
        if not text:
            self._apply_chat_mode(event)
            return

        session_key = event.unified_msg_origin or ""
        # 仅 AI 生成的文案去重；固定文案保持用户配置的原样
        dedupe = bool(self._config.get("wake_use_ai", True))
        ok = await self._send_managed(
            session_key, text,
            lambda t: event.send(event.plain_result(t)),
            dedupe=dedupe,
        )
        if ok:
            # 先发送成功，再推进冷却和沉默计时
            self._mark_wake_cooldown(event)
            self._mark_bot_spoke(session_key)
            event.should_call_llm(False)
            event.stop_event()
        else:
            self._apply_chat_mode(event)

    def _pass_probability(self, rule: WakeRule) -> bool:
        if rule.probability >= 100:
            return True
        if rule.probability <= 0:
            return False
        return random.randint(1, 100) <= rule.probability

    def _apply_chat_mode(self, event: AstrMessageEvent) -> None:
        if self.chat_mode == _MODE_ALWAYS and not self._in_quiet_time():
            event.should_call_llm(True)

    def _match_wake_rule(self, message: str) -> WakeRule | None:
        # 英文不区分大小写；wake_index 已在配置解析时按长度倒序排好
        folded = message.casefold()
        for rule in self.wake_index:
            if rule.keyword.casefold() in folded:
                return rule
        return None

    def _check_cooldown(self, event: AstrMessageEvent) -> bool:
        """只判断是否过了冷却，不消耗冷却。"""
        if self.cooldown_seconds <= 0:
            return True
        key = event.unified_msg_origin or "default"
        now = time.monotonic()
        last = self._last_wake.get(key, 0.0)
        return now - last >= self.cooldown_seconds

    def _mark_wake_cooldown(self, event: AstrMessageEvent) -> None:
        """在回复成功发送后调用，推进冷却时间戳。"""
        if self.cooldown_seconds <= 0:
            return
        key = event.unified_msg_origin or "default"
        self._last_wake[key] = time.monotonic()

    async def _make_wake_reply(
        self,
        event: AstrMessageEvent,
        rule: WakeRule,
        *,
        count: int,
        message: str,
    ) -> str:
        """count 由 _handle_message 在命中时统计好传入。"""
        has_reply = bool(rule.reply)
        has_prompt = bool(rule.prompt)

        if not has_reply and not has_prompt:
            if self.chat_mode == _MODE_ALWAYS:
                event.should_call_llm(True)
            return ""

        if not self._config.get("wake_use_ai", True):
            return rule.reply

        sender = _event_str(event, "get_sender_name") or "群友"
        ai_prompt = self._build_wake_prompt(
            rule,
            count=count,
            window=self.freq_window,
            sender=sender,
            user_text=message,
        )
        text = await self._ask_ai(event.unified_msg_origin or "", ai_prompt)
        return text or rule.reply

    @staticmethod
    def _build_wake_prompt(
        rule: WakeRule,
        *,
        count: int,
        window: int,
        sender: str,
        user_text: str,
    ) -> str:
        has_reply = bool(rule.reply)
        has_prompt = bool(rule.prompt)

        user_text = user_text.strip()
        if len(user_text) > _PROMPT_CONTEXT_LIMIT:
            user_text = user_text[:_PROMPT_CONTEXT_LIMIT] + "…"

        context = (
            f"【场景】用户「{sender}」说：{user_text}\n"
            f"【频率信息】这是关键词「{rule.keyword}」在最近 {window} 秒内"
            f"第 {count} 次被触发。\n"
        )

        if has_reply and has_prompt:
            return (
                context
                + "\n请结合用户刚才的话，按下面的要求改写一段回复文案，"
                "保持原意，不要加解释，只输出改写后的内容。\n\n"
                f"【原始文案】\n{rule.reply}\n\n"
                f"【额外要求】\n{rule.prompt}"
            )
        if has_reply:
            return (
                context
                + "\n请结合用户刚才的话，按你的系统提示词风格改写下面这段"
                "回复文案，保持原意，不要加解释，只输出改写后的内容：\n\n"
                f"{rule.reply}"
            )
        return (
            context
            + "\n请结合用户刚才的话，按下面的要求生成一条简短、自然的回复，"
            "不要加解释，只输出回复内容：\n\n"
            f"{rule.prompt}"
        )

    # ------------------------------------------------------------------
    # 频率统计与清理
    # ------------------------------------------------------------------
    def _record_and_count(self, session: str, keyword: str, window: int) -> int:
        now = time.monotonic()
        cutoff = now - window
        key = (session, keyword)

        history = self._keyword_history.get(key)
        if history is None:
            history = deque(maxlen=_MAX_HISTORY)
            self._keyword_history[key] = history
        while history and history[0] < cutoff:
            history.popleft()
        history.append(now)
        return len(history)

    def _mark_bot_spoke(self, session_id: str) -> None:
        if not session_id:
            return
        self._last_bot_speech[session_id] = time.monotonic()
        self._known_sessions.add(session_id)

    def _cleanup_stale(self, now: float) -> None:
        """清理长时间无活动的会话状态，避免无界增长。"""
        cutoff = now - _STALE_SESSION_SECONDS

        for session_id in list(self._last_bot_speech):
            if self._last_bot_speech[session_id] < cutoff:
                self._last_bot_speech.pop(session_id, None)
                self._known_sessions.discard(session_id)
                self._last_idle_chat.pop(session_id, None)
                self._recent_texts.pop(session_id, None)
                self._send_budget.pop(session_id, None)
                self._last_send_session.pop(session_id, None)

        for key in list(self._last_wake):
            if self._last_wake[key] < cutoff:
                self._last_wake.pop(key, None)

        for key in list(self._keyword_history):
            history = self._keyword_history[key]
            if not history or history[-1] < cutoff:
                self._keyword_history.pop(key, None)

    # ------------------------------------------------------------------
    # 主动发言
    # ------------------------------------------------------------------
    async def _sync_idle_task(self) -> None:
        """按当前配置启停后台主动发言任务，支持运行中热开关。"""
        enabled = bool(self._config.get("enable_idle_chat", False))
        running = self._idle_task is not None and not self._idle_task.done()

        if enabled and not running:
            await self._cancel_task(self._idle_task)
            self._idle_task = asyncio.create_task(
                self._idle_loop(), name="welcome-ai-idle"
            )
            logger.info(
                "welcome-ai 主动发言已启用：沉默 %s 秒后触发，"
                "两次主动发言最短间隔 %s 秒",
                self.idle_seconds,
                self.idle_cooldown_seconds,
            )
        elif not enabled and running:
            await self._cancel_task(self._idle_task)
            self._idle_task = None
            logger.info("welcome-ai 主动发言已关闭")

    async def _idle_loop(self) -> None:
        try:
            while True:
                # 间隔带 ±20% 随机抖动，避免固定节奏被识别成机器人
                interval = max(
                    10.0,
                    self.idle_check_interval * random.uniform(0.8, 1.2),
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
        if self._in_quiet_time():
            return

        only_group = bool(self._config.get("idle_only_group", True))
        now = time.monotonic()

        for session_id in list(self._known_sessions):
            if only_group and "GroupMessage" not in session_id:
                continue
            last_speech = self._last_bot_speech.get(session_id)
            if last_speech is None or now - last_speech < self.idle_seconds:
                continue
            # 两次主动发言之间的最短间隔
            if self.idle_cooldown_seconds > 0:
                last_idle = self._last_idle_chat.get(session_id, 0.0)
                if now - last_idle < self.idle_cooldown_seconds:
                    continue
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
            self._mark_bot_spoke(session_id)
            # AI 无输出也消耗一次 idle_cooldown，避免每轮空转反复调用 AI
            self._last_idle_chat[session_id] = time.monotonic()
            return

        ok = await self._send_managed(
            session_id, text,
            lambda t: self.context.send_message(
                session_id, MessageChain([Plain(t)])
            ),
            dedupe=True,
        )
        if not ok:
            return
        self._mark_bot_spoke(session_id)
        self._last_idle_chat[session_id] = time.monotonic()
        logger.info(
            "welcome-ai 主动发言：%s → %s", session_id, text[:40]
        )

    # ------------------------------------------------------------------
    # 命令：欢迎 @某人
    # ------------------------------------------------------------------
    @filter.command("欢迎")
    async def cmd_welcome(self, event: AstrMessageEvent, arg: str = ""):
        # 命令消息交给插件处理，避免 always_reply 模式下核心 LLM 再回一条
        event.should_call_llm(False)
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

        text = await self._make_welcome_text(
            event.unified_msg_origin or "", member_name
        )
        yield event.plain_result(text)
        event.stop_event()

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        if not self.whitelist:
            return False
        user_id = _event_str(event, "get_sender_id")
        group_id = _event_str(event, "get_group_id")
        if user_id and user_id in self.whitelist:
            return True
        if group_id and group_id in self.whitelist:
            return True
        return False

    # ------------------------------------------------------------------
    # 输出长度限制
    # ------------------------------------------------------------------
    def _enforce_length(self, text: str) -> str:
        """截断 AI 输出，避免刷屏。

        - 长度限制为 0 表示不限制
        - 按字符数截断，超出部分加省略号
        - 尽量在标点处截断，读起来更自然
        """
        limit = self.max_reply_length
        if limit <= 0:
            return text
        text = text.strip()
        if len(text) <= limit:
            return text

        # 尽量在句子结束符处断
        snippet = text[:limit]
        for punct in ("。", "！", "？", "\n", ".", "!", "?", "，", ",", "；", ";"):
            cut = snippet.rfind(punct)
            if cut >= max(1, limit // 2):
                return snippet[: cut + 1].rstrip()
        return snippet.rstrip() + "…"

    def _with_length_hint(self, prompt: str) -> str:
        """在 prompt 末尾追加长度要求，让 AI 主动遵守。"""
        limit = self.max_reply_length
        if limit <= 0:
            return prompt
        return (
            f"{prompt}\n\n"
            f"【长度要求】请把回复控制在 {limit} 个字符以内，"
            f"简短自然，不要超长。"
        )

    # ------------------------------------------------------------------
    # AI：统一入口
    # ------------------------------------------------------------------
    async def _ask_ai(self, session_id: str, prompt: str) -> str:
        # 1) 告诉 AI 遵守长度
        prompt_with_hint = self._with_length_hint(prompt)

        if self.backend == _BACKEND_PROVIDER:
            text = await self._ask_via_provider(session_id, prompt_with_hint)
        else:
            text = await self._ask_via_direct_api(prompt_with_hint)

        # 2) 兜底截断
        return self._enforce_length(text) if text else text

    # ------------------------------------------------------------------
    # AI：直连 API（带重试）
    # ------------------------------------------------------------------
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
        # 注意：0.0 是合法温度（确定性输出），不能用 `or` 兜底。
        raw_temp = self._config.get("api_temperature")
        if raw_temp is None:
            temperature = 0.8
        else:
            try:
                temperature = float(raw_temp)
            except (TypeError, ValueError):
                temperature = 0.8

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=self.api_timeout, connect=10)

        for attempt in range(_API_RETRIES + 1):
            if attempt:
                backoff = min(2.0, 0.5 * (2 ** (attempt - 1)))
                await asyncio.sleep(backoff)
            try:
                async with self._http.post(
                    url, headers=headers, json=body, timeout=timeout
                ) as resp:
                    text = await resp.text()
                    if resp.status in (429, 500, 502, 503, 504):
                        logger.warning(
                            "welcome-ai API HTTP %s，第 %d/%d 次，稍后重试：%s",
                            resp.status, attempt + 1, _API_RETRIES + 1,
                            text[:150],
                        )
                        continue
                    if resp.status >= 400:
                        logger.warning(
                            "welcome-ai API 返回 HTTP %s：%s",
                            resp.status, text[:200],
                        )
                        return ""
                    try:
                        payload = json.loads(text)
                    except Exception:
                        logger.warning(
                            "welcome-ai API 返回非 JSON：%s", text[:200]
                        )
                        return ""
                    return _extract_content(payload)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "welcome-ai API 网络错误（第 %d/%d 次）：%s",
                    attempt + 1, _API_RETRIES + 1, exc,
                )
                continue
            except Exception:
                logger.exception("welcome-ai API 调用失败")
                return ""

        logger.warning(
            "welcome-ai API 重试 %d 次后仍失败，放弃", _API_RETRIES + 1
        )
        return ""

    # ------------------------------------------------------------------
    # AI：复用 AstrBot provider
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_call(func: Callable[[], Any]) -> Any:
        try:
            return func()
        except Exception:
            return None

    def _get_provider(self, session_id: str):
        preferred = str(self._config.get("preferred_provider_id") or "").strip()
        if preferred:
            provider = self._safe_call(
                lambda: self.context.get_provider_by_id(pid=preferred)
            )
            if provider is not None:
                return provider

        pid = self._safe_call(
            lambda: self.context.get_current_chat_provider_id(umo=session_id)
        )
        if pid:
            provider = self._safe_call(
                lambda: self.context.get_provider_by_id(pid=pid)
            )
            if provider is not None:
                return provider

        provider = self._safe_call(
            lambda: self.context.get_using_provider(umo=session_id)
        )
        if provider is not None:
            return provider
        provider = self._safe_call(
            lambda: self.context.get_using_provider()
        )
        if provider is not None:
            return provider

        providers = self._safe_call(
            lambda: self.context.get_all_providers()
        )
        if providers:
            return providers[0]
        return None

    async def _ask_via_provider(self, session_id: str, prompt: str) -> str:
        provider = self._get_provider(session_id)
        if provider is None:
            logger.warning("welcome-ai 找不到 AstrBot provider，跳过 AI 调用")
            return ""

        chat_session_id = f"welcome-ai-{session_id}"

        # 用 inspect 检查 text_chat 的签名来决定传哪些参数，
        # 不用 except TypeError —— 那会把 provider 内部真正的类型错误吞掉。
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "session_id": chat_session_id,
        }
        try:
            sig = inspect.signature(provider.text_chat)
            params = sig.parameters
            # 有 **kwargs 时也认为是兼容的
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            if has_var_kw or "persist" in params:
                kwargs["persist"] = False
            if has_var_kw or "system_prompt" in params:
                kwargs["system_prompt"] = self.system_prompt
        except (TypeError, ValueError):
            # 拿不到签名：保守一点，只传最基础的参数
            pass

        try:
            resp = await provider.text_chat(**kwargs)
        except Exception:
            logger.exception("welcome-ai provider.text_chat 失败")
            return ""

        text = getattr(resp, "completion_text", "") or ""
        return str(text).strip()

    def _log_provider_list(self) -> None:
        providers = self._safe_call(
            lambda: self.context.get_all_providers()
        )
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
