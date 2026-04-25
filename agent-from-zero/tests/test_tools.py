from pathlib import Path

from agent_from_zero.tools.calculator import calculate_expression
from agent_from_zero.tools.report_writer import write_report


def test_calculator() -> None:
    assert calculate_expression("18 * 27") == 486


def test_report_writer() -> None:
    path = write_report("pytest_report", "hello")
    assert Path(path).exists()
    assert "hello" in Path(path).read_text(encoding="utf-8")
