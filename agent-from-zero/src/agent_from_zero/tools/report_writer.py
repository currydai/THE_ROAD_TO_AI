from pathlib import Path
from re import sub

from langchain_core.tools import tool

from agent_from_zero.config.settings import get_settings


def write_report(title: str, content: str) -> Path:
    settings = get_settings()
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    safe_title = sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", title).strip("_") or "report"
    path = settings.report_dir / f"{safe_title}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return path


@tool
def write_markdown_report(title: str, content: str) -> str:
    """把 Markdown 报告写入安全的报告目录，并返回文件路径。"""
    return str(write_report(title, content))
