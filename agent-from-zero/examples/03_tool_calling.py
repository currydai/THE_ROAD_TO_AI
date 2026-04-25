from agent_from_zero.tools.calculator import calculator
from agent_from_zero.tools.report_writer import write_markdown_report


if __name__ == "__main__":
    value = calculator.invoke({"expression": "18 * 27"})
    path = write_markdown_report.invoke(
        {"title": "tool_calling_result", "content": f"18 * 27 = {value}"}
    )
    print(f"计算结果：{value}")
    print(f"报告路径：{path}")
