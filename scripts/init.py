import argparse
import subprocess

from hand_to_tex.utils import logger

PREPROCESS_EXTENDED_PARAMS = [
    "--out-dir",
    "data/extended",
    "--merge",
    "synthetic",
    "symbols",
]
PREPROCESS_DEFAULT_PARAMS = [
    "--out-dir",
    "data/full",
]
PREPROCESS_MOCK_PARAMS = [
    "--root",
    "data/mathwriting-2024-excerpt",
    "--out-dir",
    "data/sample",
    "--merge",
    "synthetic",
    "symbols",
]
INIT_MODES = ["mock", "standard", "extended"]


def run_init(threads: int, mode: str = "standard") -> None:
    match mode:
        case "mock":
            commands = [
                ["htt-get-data"],
                ["htt-preprocess", "--threads", str(threads)] + PREPROCESS_MOCK_PARAMS,
            ]
        case "standard":
            commands = [
                ["htt-get-data", "--full"],
                ["htt-preprocess", "--threads", str(threads)] + PREPROCESS_DEFAULT_PARAMS,
            ]
        case "extended":
            commands = [
                ["htt-get-data", "--full"],
                ["htt-preprocess", "--threads", str(threads)] + PREPROCESS_EXTENDED_PARAMS,
            ]
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    for command in commands:
        logger.info(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize dataset and run preprocessing in one command",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker processes passed to htt-preprocess (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=INIT_MODES,
        default="standard",
        help=(
            "Initialization mode:\n"
            "  mock (~1.5 MB): excerpt dataset + preprocess to data/sample\n"
            "  standard (~2 GB): full dataset + preprocess to data/full\n"
            "  extended (~5 GB): full dataset + preprocess to data/extended with synthetic+symbols merge"
        ),
    )
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error(f"Argument --threads must be > 0 (got {args.threads})")

    try:
        run_init(threads=args.threads, mode=args.mode)
    except FileNotFoundError as exc:
        logger.error(f"Required CLI command not found: {exc}")
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        logger.error(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        raise SystemExit(exc.returncode or 1) from exc


if __name__ == "__main__":
    main()
