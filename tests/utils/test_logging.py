"""Tests for the logging utility.

Verifies that the package-level logger is correctly exported and that
log messages are properly captured by our test fixtures.
"""

from __future__ import annotations

from typing import Any

from loguru import logger as loguru_logger

from hand_to_tex.utils import logger
from hand_to_tex.utils.logging import logger as logger_direct


class TestLoggerExport:
    """Test suite for logger configuration and export."""

    def test_logger_is_the_loguru_logger(self) -> None:
        """The package logger must be an instance of the loguru logger."""
        assert logger is loguru_logger

    def test_direct_and_package_imports_match(self) -> None:
        """Direct and indirect logger imports must point to the same object."""
        assert logger is logger_direct

    def test_logger_records_messages(self, caplog_loguru: Any) -> None:
        """The caplog_loguru fixture must correctly capture logger output."""
        logger.info("hello-from-test")
        assert "hello-from-test" in caplog_loguru.text
