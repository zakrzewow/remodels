"""Pytest conf file."""

from datetime import datetime

import pytest


def pytest_html_results_table_header(cells):
    """Conf function that adds 'Description' column."""
    del cells[-1]
    cells.insert(1, "<th>Description</th>")


def pytest_html_results_table_row(report, cells):
    """Conf function that adds 'Description' rows."""
    del cells[-1]
    cells.insert(1, f"<td>{report.description}</td>")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Conf function that adds 'Description' rows content."""
    outcome = yield
    report = outcome.get_result()
    report.description = str(item.function.__doc__)
