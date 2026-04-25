from pathlib import Path

from langchain_core.tools import tool

from agent_from_zero.config.settings import get_settings


def _safe_path(path: str) -> Path:
    settings = get_settings()
    root = settings.allowed_file_root.resolve()
    target = Path(path).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"文件路径必须位于 {root} 下")
    return target


def read_text(path: str) -> str:
    target = _safe_path(path)
    if target.suffix.lower() not in {".txt", ".md", ".csv"}:
        raise ValueError("当前教程只允许读取 .txt/.md/.csv")
    return target.read_text(encoding="utf-8")


@tool
def read_text_file(path: str) -> str:
    """读取 data 目录下的文本、Markdown 或 CSV 文件。"""
    return read_text(path)
