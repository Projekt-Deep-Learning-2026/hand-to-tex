import argparse
import subprocess

from hand_to_tex.utils import logger

PREPROCESS_DEFAULT_PARAMS = [
    "--out-dir",
    "data/full",
    "--merge",
    "synthetic",
    "symbols",
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


def run_init(threads: int, mock: bool = False) -> None:
    if mock:
        commands = [
            ["htt-get-data"],
            ["htt-preprocess", "--threads", str(threads)] + PREPROCESS_MOCK_PARAMS,
        ]
    else:
        commands = [
            ["htt-get-data", "--full"],
            ["htt-preprocess", "--threads", str(threads)] + PREPROCESS_DEFAULT_PARAMS,
        ]

    for command in commands:
        logger.info(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize full dataset and run preprocessing in one command"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker processes passed to htt-preprocess (default: 1)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use sample dataset (excerpt) and mock preprocessing defaults",
    )
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error(f"Argument --threads must be > 0 (got {args.threads})")

    try:
        run_init(threads=args.threads, mock=args.mock)
    except FileNotFoundError as exc:
        logger.error(f"Required CLI command not found: {exc}")
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        logger.error(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        raise SystemExit(exc.returncode or 1) from exc


if __name__ == "__main__":
    main()
