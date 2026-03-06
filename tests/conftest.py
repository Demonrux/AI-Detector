import pytest
from pathlib import Path
from _pytest.reports import TestReport
from datetime import datetime


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        log_file = Path(__file__).parent.parent / "logs" / "test.log"
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{report.nodeid} - {report.outcome}\n")
            if report.longrepr:
                f.write(f"  {report.longrepr}\n")


def pytest_sessionstart(session) -> None:
    """Вызывается перед началом тестовой сессии"""
    log_file = Path(__file__).parent.parent / "logs" / "test.log"
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"ЗАПУСК ТЕСТОВ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")