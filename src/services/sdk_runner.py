"""
Claude Agent SDK 调用封装

- 工作目录：data/tasks/<sanitize(linkid)>/
- 支持 session 复用：通过 resume 参数恢复历史对话
- CLAUDE.md 从 prompts/claude-md.md 模板动态生成，仅首次写入
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

from ..config import Config
from ..logger import get_logger
from .sse_bus import publish

log = get_logger()


def _sanitize_linkid(s: str) -> str:
    """替换路径分隔符和危险字符，确保 mkdir 成功。"""
    return re.sub(r'[/\\:*?"<>|]', '_', s)


def _deep_get(obj: dict, path: str) -> Any:
    parts = path.split(".")
    cur: Any = obj
    for p in parts:
        if cur is None or not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _build_env(cfg: Config) -> dict[str, str | None]:
    """根据 LLM 配置构建子进程环境变量。"""
    env: dict[str, str | None] = {}
    llm = cfg.llm

    env["ANTHROPIC_BASE_URL"] = llm.base_url

    if llm.auth_token:
        env["ANTHROPIC_AUTH_TOKEN"] = llm.auth_token
    else:
        env["ANTHROPIC_AUTH_TOKEN"] = ""

    if llm.model:
        env["ANTHROPIC_MODEL"] = llm.model
    if llm.opus_model:
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = llm.opus_model
    if llm.sonnet_model:
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = llm.sonnet_model
    if llm.haiku_model:
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = llm.haiku_model
    if llm.subagent_model:
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = llm.subagent_model
    if llm.effort_level:
        env["CLAUDE_CODE_EFFORT_LEVEL"] = llm.effort_level
    if llm.timeout_ms:
        env["API_TIMEOUT_MS"] = str(llm.timeout_ms)

    # 文件下载 Token（优先从 CSV prompt_config 读取）
    from ..store import prompt_config_store
    file_url_token = (
        prompt_config_store.get_plain("doc.file_url_token")
        or prompt_config_store.get_plain("sys.file_url_token")
        or cfg.sys.file_url_token
    )
    if file_url_token:
        env["FILE_URL_TOKEN"] = file_url_token
    else:
        env["FILE_URL_TOKEN"] = ""

    # 文档审核服务 Token
    doc_audit_token = (
        prompt_config_store.get_plain("doc_audit.token")
        or cfg.doc_audit.token
    )
    if doc_audit_token:
        env["DOC_AUDIT_TOKEN"] = doc_audit_token
    else:
        env["DOC_AUDIT_TOKEN"] = ""

    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1" if llm.disable_nonessential_traffic else "0"
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "1" if llm.attribution_header else "0"
    env["CLAUDE_CODE_AUTO_MODE_SERVER"] = "1" if llm.auto_mode_server else "0"

    return env


def _load_all_prompt_files() -> dict[str, str]:
    """加载 prompts/ 目录下的所有模板文件（含 sections/）。"""
    files: dict[str, str] = {}
    prompts_dir = Path.cwd() / "prompts"
    if not prompts_dir.exists():
        return files

    def walk(d: Path, prefix: str) -> None:
        if not d.exists():
            return
        for entry in sorted(d.iterdir()):
            full = d / entry.name
            rel = prefix + entry.name
            if entry.is_dir():
                walk(full, rel + "/")
            elif entry.name.endswith(".md"):
                try:
                    files[rel] = full.read_text(encoding="utf-8")
                except Exception:
                    pass

    walk(prompts_dir, "")
    return files


def _load_prompt_file(relative_path: str) -> str | None:
    """从 prompts 目录加载模板文件。"""
    candidates = [
        Path.cwd() / "prompts",
    ]
    for d in candidates:
        p = d / relative_path
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return None


async def _build_claude_md(cfg: Config) -> str:
    """从 prompts/claude-md.md 模板 + CSV 配置 + sections 构建 CLAUDE.md。"""
    template = _load_prompt_file("claude-md.md") or ""
    if not template:
        # 兜底默认模板
        template = f"""# dp-agt-py 工作区

你是电子公文处理智能体，运行在这个临时工作目录中。

## 文件下载
- 对方传递的 JSON 中的文件 URL 已开放白名单，无需认证
- 如果文件 URL 需要认证，可用环境变量 FILE_URL_TOKEN（Bearer Token）下载：
  `curl -s -k --noproxy '*' -w'\\n' -H "Authorization: Bearer $FILE_URL_TOKEN" '<file_url>' -o 文件名`
- 用 curl 下载文件到当前工作目录

## 文档解析服务

同网络下有一个文档解析服务，可将 PDF / DOCX / .doc（旧版 Word）转为纯文本，
同时自动检测印章。

```bash
curl -s -k --noproxy '*' -w'\\n' -X POST {cfg.sys.doc_util_url}/extract/txt \\
  -F "file=@文件名" | jq '.'
```

返回 JSON:
- text: 提取的全部文本
- has_seal: 是否检测到印章 (true/false)
- seal_text: 印章文字列表
- ocr_confidence: OCR 置信度 0.0–1.0

## 行为规范
1. 所有文本内容必须使用简体中文
2. 严格按提示词要求的 JSON Schema 输出
3. 最终结果用 ```json 代码块包裹
4. json 代码块之外不要输出任何其他内容
5. 下载失败或文件无法识别，在 JSON 中标记 need_human: true
6. 如果请求涉及某个文件但上下文里没有其文本内容，按"文件下载"章节自行 curl 下载"""
        return template

    # ── 从 prompt_config（CSV）+ cfg.yml 系统级收集所有值 ──
    from ..store import prompt_config_store
    cfg_vals = prompt_config_store.get_all_plain()

    # 系统级占位符（不在 CSV 中）
    cfg_vals["server.port"] = str(cfg.server.port)
    cfg_vals["server.self_ip"] = str(cfg.server.self_ip or "")
    cfg_vals["sys.doc_util_url"] = cfg.sys.doc_util_url
    cfg_vals["doc_util_url"] = cfg.sys.doc_util_url

    # ── 预加载所有 sections 文件 ──
    all_sections = _load_all_prompt_files()

    # ── 替换 {{section:name:trigger_key}} 占位符 ──
    def _replace_section(m: re.Match[str]) -> str:
        name = m.group(1)
        trigger_key = m.group(2).strip()
        trigger_val = cfg_vals.get(trigger_key)
        if not trigger_val:
            return ""  # 触发 key 没值 → 整段不渲染
        section_key = "sections/" + name + ".md"
        section_content = all_sections.get(section_key)
        if section_content is None:
            section_content = all_sections.get(name + ".md")
        # sections 内部也有占位符，需要再做一轮替换
        if section_content:
            section_content = re.sub(
                r"\{\{([\w.]+)\}\}",
                lambda m2: cfg_vals.get(m2.group(1).strip(), m2.group(0)),
                section_content,
            )
        return section_content or ""

    template = re.sub(
        r"\{\{section:(\w+):([\w.]+)\}\}",
        _replace_section,
        template,
    )

    # ── 替换剩余 {{key}} 占位符（key 支持含点号） ──
    template = re.sub(
        r"\{\{([\w.]+)\}\}",
        lambda m: cfg_vals.get(m.group(1).strip(), m.group(0)),
        template,
    )

    return template


@dataclass
class SdkResult:
    success: bool
    raw_output: str
    trace: str
    session_id: str | None = None  # SDK 返回的 session_id，用于后续 resume
    error_message: str | None = None
    duration_ms: int = 0


async def run_agent(
    linkid: str,
    task_id: str,
    rendered_prompt: str,
    cfg: Config,
    resume_session_id: str | None = None,
    on_session_id: Callable[[str], None] | None = None,
) -> SdkResult:
    """运行 Claude Agent SDK 处理一个任务。

    Args:
        linkid: 公文审批 id，同时作为工作目录名（sanitize 后）
        rendered_prompt: 渲染后的提示词
        cfg: 配置
        resume_session_id: 若传入则 resume 该 session（同一公文复用会话）
        on_session_id: 首条流式消息拿到 session_id 的瞬间回调（用于立即写 DB，让前端实时看到会话）
    """
    work_dir = Path(cfg.sys.data_dir) / "tasks" / _sanitize_linkid(linkid)
    work_dir.mkdir(parents=True, exist_ok=True)

    # CLAUDE.md 仅首次写入（resume 时不覆盖，因为 SDK snapshot 已固化 system prompt）
    if not resume_session_id:
        (work_dir / "CLAUDE.md").write_text(
            await _build_claude_md(cfg),
            encoding="utf-8",
        )

    log.info(
        "sdk_run_start task_id=%s linkid=%s resume_session_id=%s work_dir=%s",
        task_id, linkid, resume_session_id or "(new)", work_dir,
    )

    start = time.monotonic()
    full_output = ""
    captured_session_id: str | None = None
    trace_messages: list[Any] = []

    try:
        # 应用 LLM 环境变量
        env_overrides = _build_env(cfg)
        for k, v in env_overrides.items():
            if v:
                os.environ[k] = v

        options = ClaudeAgentOptions(
            model=cfg.llm.model,
            cwd=str(work_dir),
            allowed_tools=["Bash", "Read", "Write", "Glob", "Grep", "WebSearch", "WebFetch"],
            permission_mode="dontAsk",
            system_prompt=await _build_claude_md(cfg),
            max_buffer_size=10 * 1024 * 1024,
            **({"resume": resume_session_id} if resume_session_id else {}),
        )

        async for message in query(prompt=rendered_prompt, options=options):
            trace_messages.append(message)

            # 首条消息即捕获 session_id，用于立即写入 sessions 表
            sid = getattr(message, "session_id", None)
            if not captured_session_id and isinstance(sid, str) and sid:
                captured_session_id = sid
                if on_session_id:
                    on_session_id(sid)

            # 推送到 SSE 总线，让前端实时订阅
            if captured_session_id:
                publish(captured_session_id, message)

            # 收集 assistant 文本输出
            if isinstance(message, AssistantMessage):
                if message.content:
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            full_output += block.text

        duration_ms = int((time.monotonic() - start) * 1000)
        trace = json.dumps(trace_messages, default=str, ensure_ascii=False)
        log.info(
            "sdk_run_complete task_id=%s linkid=%s session_id=%s duration_ms=%d output_len=%d trace_len=%d",
            task_id, linkid, captured_session_id, duration_ms, len(full_output), len(trace),
        )

        # 发送结束信号，让 SSE 连接自然关闭
        if captured_session_id:
            publish(captured_session_id, {"type": "end"})

        return SdkResult(
            success=True,
            raw_output=full_output,
            trace=trace,
            session_id=captured_session_id,
            duration_ms=duration_ms,
        )

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        message = str(exc)
        trace = json.dumps(trace_messages, default=str, ensure_ascii=False)
        log.error(
            "sdk_run_error task_id=%s linkid=%s session_id=%s duration_ms=%d error=%s",
            task_id, linkid, captured_session_id, duration_ms, message,
        )

        if captured_session_id:
            publish(captured_session_id, {"type": "end"})
        return SdkResult(
            success=False,
            raw_output=full_output,
            trace=trace,
            session_id=captured_session_id,
            error_message=message,
            duration_ms=duration_ms,
        )