import sys
from unittest.mock import call, patch

import pytest

from scripts import init


def test_run_init_executes_commands_in_order():
    with patch("scripts.init.subprocess.run") as run_mock:
        init.run_init(threads=12, mock=False)

    assert run_mock.call_args_list == [
        call(["htt-get-data", "--full"], check=True),
        call(
            ["htt-preprocess", "--threads", "12"] + init.PREPROCESS_DEFAULT_PARAMS,
            check=True,
        ),
    ]


def test_run_init_executes_mock_commands_in_order():
    with patch("scripts.init.subprocess.run") as run_mock:
        init.run_init(threads=3, mock=True)

    assert run_mock.call_args_list == [
        call(["htt-get-data"], check=True),
        call(["htt-preprocess", "--threads", "3"] + init.PREPROCESS_MOCK_PARAMS, check=True),
    ]


def test_main_passes_threads_to_run_init(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["htt-init", "--threads", "7"])
    captured = {}

    def fake_run_init(threads: int, mock: bool) -> None:
        captured["threads"] = threads
        captured["mock"] = mock

    monkeypatch.setattr(init, "run_init", fake_run_init)
    init.main()

    assert captured["threads"] == 7
    assert captured["mock"] is False


def test_main_passes_mock_flag_to_run_init(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["htt-init", "--threads", "2", "--mock"])
    captured = {}

    def fake_run_init(threads: int, mock: bool) -> None:
        captured["threads"] = threads
        captured["mock"] = mock

    monkeypatch.setattr(init, "run_init", fake_run_init)
    init.main()

    assert captured["threads"] == 2
    assert captured["mock"] is True


def test_main_rejects_non_positive_threads(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["htt-init", "--threads", "0"])

    with pytest.raises(SystemExit):
        init.main()
