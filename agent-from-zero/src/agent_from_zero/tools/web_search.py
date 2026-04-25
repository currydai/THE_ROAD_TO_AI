from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """占位搜索工具。生产项目中可替换为 Tavily、SerpAPI 或企业搜索。"""
    return f"示例搜索结果：你搜索了 {query}。"
