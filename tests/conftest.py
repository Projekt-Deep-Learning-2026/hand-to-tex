from pathlib import Path

import pytest

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
def vocab() -> LatexVocab:
    return LatexVocab.default()
