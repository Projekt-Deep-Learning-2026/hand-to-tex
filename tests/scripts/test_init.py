import sys
from unittest.mock import call, patch

import pytest

from scripts import init


def test_run_init_executes_standard_commands_in_order():
    with patch("scripts.init.subprocess.run") as run_mock:
        init.run_init(threads=12, mode="standard")

    assert run_mock.call_args_list == [
        call(["htt-get-data", "--full"], check=True),
        call(
            ["htt-preprocess", "--threads", "12"] + init.PREPROCESS_DEFAULT_PARAMS,
            check=True,
        ),
    ]


def test_run_init_executes_mock_commands_in_order():
    with patch("scripts.init.subprocess.run") as run_mock:
        init.run_init(threads=3, mode="mock")

    assert run_mock.call_args_list == [
        call(["htt-get-data"], check=True),
        call(["htt-preprocess", "--threads", "3"] + init.PREPROCESS_MOCK_PARAMS, check=True),
    ]


def test_run_init_executes_extended_commands_in_order():
    with patch("scripts.init.subprocess.run") as run_mock:
        init.run_init(threads=5, mode="extended")

    assert run_mock.call_args_list == [
        call(["htt-get-data", "--full"], check=True),
        call(
            ["htt-preprocess", "--threads", "5"] + init.PREPROCESS_EXTENDED_PARAMS,
            check=True,
        ),
    ]


def test_main_passes_threads_to_run_init(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["htt-init", "--threads", "7"])
    captured = {}

    def fake_run_init(threads: int, mode: str) -> None:
        captured["threads"] = threads
        captured["mode"] = mode

    monkeypatch.setattr(init, "run_init", fake_run_init)
    init.main()

    assert captured["threads"] == 7
    assert captured["mode"] == "standard"


def test_main_passes_mode_to_run_init(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["htt-init", "--threads", "2", "--mode", "mock"])
    captured = {}

    def fake_run_init(threads: int, mode: str) -> None:
        captured["threads"] = threads
        captured["mode"] = mode

    monkeypatch.setattr(init, "run_init", fake_run_init)
    init.main()

    assert captured["threads"] == 2
    assert captured["mode"] == "mock"


def test_main_rejects_non_positive_threads(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["htt-init", "--threads", "0"])

    with pytest.raises(SystemExit):
        init.main()
