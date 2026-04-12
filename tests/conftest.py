from pathlib import Path

import pytest
from loguru import logger

from hand_to_tex.utils import LatexVocab


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_inkml(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.inkml"


@pytest.fixture
def minimal_symbols_inkml(fixtures_dir: Path) -> Path:
    return fixtures_dir / "minimal_symbols.inkml"


@pytest.fixture
def caplog_loguru():
    """Fixture to capture loguru logs in a list."""
    handler_id = None
    messages = []

    def capture_handler(message):
        messages.append(message.record["message"])

    handler_id = logger.add(capture_handler, format="{message}")

    class LogCapture:
        @property
        def text(self):
            return "\n".join(messages)

    yield LogCapture()
    if handler_id is not None:
        logger.remove(handler_id)


@pytest.fixture
def vocab() -> LatexVocab:
    return LatexVocab.default()
