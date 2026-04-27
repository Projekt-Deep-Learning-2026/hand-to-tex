from __future__ import annotations

from loguru import logger as loguru_logger

from hand_to_tex.utils import logger
from hand_to_tex.utils.logging import logger as logger_direct


class TestLoggerExport:
    def test_logger_is_the_loguru_logger(self):
        assert logger is loguru_logger

    def test_direct_and_package_imports_match(self):
        assert logger is logger_direct

    def test_logger_records_messages(self, caplog_loguru):
        logger.info("hello-from-test")
        assert "hello-from-test" in caplog_loguru.text
