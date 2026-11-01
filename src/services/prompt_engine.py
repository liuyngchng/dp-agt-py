"""
提示词渲染引擎 — 支持 `{{request.body}}` `{{request.linkid}}` `{{request.xxx.yyy}}` 等多层级占位符。
从 prompts/ 目录加载模板文件，支持文件发现 + CSV 占位符替换 + section 条件渲染。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{\{(request\.[^}]+)\}\}")
_PROMPT_MAP: dict[str, PromptFile] = {}


@dataclass
class PromptFile:
    filename: str   # 相对路径，如 doc/classify.md
    path: str       # 访问路径，如 doc/classify
    name: str       # 从文件中提取的标题（第一行 #），否则用 filename
    content: str | None = None


def _deep_get(obj: Any, path: str) -> Any:
    """从对象中按点号路径取值。"""
    parts = path.split(".")
    cur: Any = obj
    for p in parts:
        if cur is None or not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def render_prompt(template: str, request_body: dict[str, Any]) -> str:
    """渲染提示词，替换 {{placeholder}}。"""

    def _replace(match: re.Match[str]) -> str:
        path = match.group(1)
        field_path = path[len("request."):]

        if field_path == "body":
            value: Any = request_body
        else:
            value = _deep_get(request_body, field_path)

        if value is None:
            return ""  # 找不到的字段用空字符串替换
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2)

    return _PLACEHOLDER_RE.sub(_replace, template)


def rescan_prompts_dir() -> list[PromptFile]:
    """重新扫描 prompts/ 目录，更新 path→file 映射。"""
    global _PROMPT_MAP
    _PROMPT_MAP.clear()
    results: list[PromptFile] = []

    dirs = [
        Path.cwd() / "prompts",
    ]

    for base_dir in dirs:
        if not base_dir.exists():
            continue

        def walk(d: Path, prefix: str) -> None:
            for entry in sorted(d.iterdir()):
                if entry.name.startswith("_") or entry.name == "sections":
                    continue
                rel = prefix + entry.name
                if entry.is_dir():
                    walk(entry, rel + "/")
                elif entry.is_file() and entry.name.endswith(".md"):
                    rel_path = rel.replace(".md", "")  # doc/classify
                    basename = entry.name.replace(".md", "")

                    # claude-md 不作为访问路径
                    if basename == "claude-md":
                        continue

                    name = basename
                    content: str | None = None
                    try:
                        content = entry.read_text(encoding="utf-8")
                        m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
                        if m:
                            name = m.group(1).strip()
                    except Exception:
                        pass

                    pf = PromptFile(filename=rel, path=rel_path, name=name, content=content)
                    _PROMPT_MAP[rel_path] = pf
                    results.append(pf)

        walk(base_dir, "")
        break

    return results


def load_prompt_from_disk(path: str) -> str | None:
    """从 prompts/ 目录读取指定 path 对应的提示词。"""
    pf = _PROMPT_MAP.get(path)
    if pf and pf.content is not None:
        return pf.content

    # 兜底：path 如 "doc/classify" → 文件 "prompts/doc/classify.md"
    filename = path + ".md"
    dirs = [
        Path.cwd() / "prompts",
    ]
    for d in dirs:
        p = d / filename
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return None


def list_all_paths() -> list[str]:
    return list(_PROMPT_MAP.keys())


def list_all_prompt_files() -> list[PromptFile]:
    return list(_PROMPT_MAP.values())


def path_to_name(path: str) -> str:
    pf = _PROMPT_MAP.get(path)
    return pf.name if pf else path