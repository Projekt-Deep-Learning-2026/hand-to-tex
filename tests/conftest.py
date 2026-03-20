from pathlib import Path
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def sample_inkml(fixtures_dir: Path) -> Path:
    return fixtures_dir / 'sample.inkml'


@pytest.fixture
def minimal_symbols_inkml(fixtures_dir: Path) -> Path:
    return fixtures_dir / 'minimal_symbols.inkml'
