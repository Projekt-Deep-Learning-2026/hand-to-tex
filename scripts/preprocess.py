import argparse
import concurrent.futures
import itertools
import random
import shutil
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from hand_to_tex.datasets import HMEDatasetRaw, InkData
from hand_to_tex.utils import LatexVocab, logger

BATCH_SIZE = 1000
SPLITS = ["train", "valid", "test"]
SPLIT_T = Literal["train", "valid", "test"]
MERGE_SPLIT_SEED = 42
MERGE_WEIGHTS = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}
DATASET_PATH = Path("./data/mathwriting-2024")
VOCAB_PATH = Path("./data/assets/vocab.json")


def _process_single_file(
    pth: Path,
    vocab: LatexVocab,
    max_tokens: None | int,
    max_tracepoints: None | int,
) -> tuple[
    Literal["success", "error", "empty", "filtered"], tuple[torch.Tensor, torch.Tensor] | None, str
]:
    """Process single .inkml file, perform validation and return status and pair (features, tokens)"""
    ID = pth.stem
    ERR = ("error", None, ID)
    EMPTY = ("empty", None, ID)
    FILTERED = ("filtered", None, ID)
    try:
        ink = InkData.load(pth)
        fts = HMEDatasetRaw.extract_features(ink)

        if fts.size(0) == 0:
            return EMPTY

        tokens = vocab.encode_expr(ink.tex_norm)
        tokens = torch.tensor(tokens, dtype=torch.long)

        if (
            (max_tracepoints is not None and fts.size(0) > max_tracepoints)
            or (max_tokens is not None and tokens.size(0) > max_tokens)
            or (vocab.UNK in tokens)
        ):
            return FILTERED

        return "success", (fts, tokens), ID

    except Exception as _e:
        return ERR


def preprocess_split(
    root: Path,
    out_dir: Path,
    split_name: str,
    vocab: LatexVocab,
    num_workers: int,
    start_idx: int,
    capacity: int | None,
    max_tokens: int | None,
    max_tracepoints: int | None,
):
    split_dir = root / split_name
    if not split_dir.exists():
        logger.error(f"Directory {split_dir} not found")
        return

    if not out_dir.exists():
        logger.info(f"Output directory {out_dir} not found, creating")
        out_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = Path(out_dir, split_name + ".pt")

    inkmls = sorted(split_dir.rglob("*.inkml"))
    if len(inkmls) <= start_idx:
        logger.warning(f"No .inkml files found in {split_dir} (after skipping {start_idx} files)")
        return

    inkmls = inkmls[start_idx:]

    if capacity is None:
        capacity = len(inkmls)

    logger.info(
        f"Processing split {split_name}, {len(inkmls)} .inkml files found, using {num_workers} workers"
    )

    total_saved = 0
    temp_files = []
    empty_cnt, err_cnt, filter_cnt = 0, 0, 0
    try:
        with (
            concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor,
            tqdm(total=len(inkmls), desc=split_name) as pbar,
        ):
            for b_idx, batch_inkmls in enumerate(itertools.batched(inkmls, BATCH_SIZE)):
                current = {}
                futures = {
                    executor.submit(
                        _process_single_file,
                        pth,
                        vocab,
                        max_tokens,
                        max_tracepoints,
                    ): pth
                    for pth in batch_inkmls
                }
                for future in concurrent.futures.as_completed(futures):
                    status, res, ID = future.result()
                    match status:
                        case "success":
                            assert res is not None, (
                                "When status=success res should be (Tensor, Tensor), not None"
                            )
                            current[ID] = res
                        case "empty":
                            empty_cnt += 1
                        case "error":
                            err_cnt += 1
                        case "filtered":
                            filter_cnt += 1
                    pbar.update(1)

                remaining = capacity - total_saved

                to_save = [v for _k, v in sorted(current.items())][:remaining]
                temp_path = out_dir / f"{split_name}_temp_{b_idx}.pt"
                torch.save(to_save, temp_path)
                temp_files.append(temp_path)

                total_saved += len(to_save)
                pbar.set_postfix(gathered=str(total_saved))

                if total_saved >= capacity:
                    pbar.set_description(f"Done, capacity={capacity} achieved")
                    break

        final_data = []
        logger.info(f"Saving to {out_file_path}")
        for temp in temp_files:
            final_data.extend(torch.load(temp, weights_only=True))

        torch.save(final_data, out_file_path)

        logger.info(f"Preprocessing {split_name}, summary:")
        logger.info(f"Saved: {len(final_data)} out of {len(inkmls)}")
        logger.info(f"Empty samples: {empty_cnt}")
        logger.info(f"Error samples: {err_cnt}")
        logger.info(f"Filtered-out samples: {filter_cnt}")
        logger.info("=" * 50)

    finally:
        for temp in tqdm(temp_files, "Cleaning temp files"):
            temp.unlink()
        logger.info(f"Cleaned {len(temp_files)} temp files succesfully")


def merge_into(
    merge_data: list[tuple[torch.Tensor, torch.Tensor]], out_dir: Path, splits: list[SPLIT_T]
):

    shuffled = list(merge_data)
    rng = random.Random(MERGE_SPLIT_SEED)
    rng.shuffle(shuffled)

    total = len(shuffled)
    total_weight = sum(MERGE_WEIGHTS[spl] for spl in splits)
    normalized_weights = {spl: MERGE_WEIGHTS[spl] / total_weight for spl in splits}

    counts: dict[SPLIT_T, int] = {spl: int(total * normalized_weights[spl]) for spl in splits}
    used = sum(counts.values())
    if used < total:
        remainder = total - used
        ordered = sorted(
            splits,
            key=lambda spl: (total * normalized_weights[spl] - counts[spl], SPLITS.index(spl)),
            reverse=True,
        )
        for i in range(remainder):
            counts[ordered[i % len(ordered)]] += 1

    data_to_append: dict[SPLIT_T, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    left = 0
    for spl in splits:
        right = left + counts[spl]
        data_to_append[spl] = shuffled[left:right]
        left = right

    for spl in splits:
        split_path = Path(out_dir, spl + ".pt")
        d = torch.load(split_path, weights_only=True)
        d.extend(data_to_append[spl])
        torch.save(d, split_path)

    summary = ", ".join(f"{spl}={counts[spl]}" for spl in splits)
    logger.info(f"Merged samples summary: total={total} ({summary})")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-processing of dataset Hand-to-TeX")
    parser.add_argument(
        "--root",
        type=str,
        required=False,
        default=DATASET_PATH,
        help=f"Path to dataset containing all splits with .inkml files, default={DATASET_PATH}",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=False,
        default=VOCAB_PATH,
        help=f"Path to .json file containing vocabulary mapping for latex symbols, default={VOCAB_PATH}",
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Number of workers (processes) to use for preprocessing, default=1",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=str,
        required=False,
        choices=SPLITS,
        default=SPLITS,
        help=f"List of splits to preprocess, choices are: {', '.join(SPLITS)}, by default all splits will be preprocessed",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        help="Directory where .pt preprocessed splits are going to be saved, by default the `root` directory will be used",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        required=False,
        help="While preprocessing splits first `start_idx` files will be omitted, helpful for generating partial datasets",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        required=False,
        help="At most `capacity` files are going to be saved, helpful for creating partial datasets",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=False,
        help="Setting this parameter enables you to omit samples that have more than this many tokens in normalised form. Alias: --max-len",
    )
    parser.add_argument(
        "--max-tracepoints",
        type=int,
        required=False,
        help="Setting this parameter enables you to omit samples that have more than this many tracepoints in feature representation",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="List of folders in `--root` to merge into splits that will be created in current run",
    )

    return parser


def validate_parser(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    if args.capacity is not None and args.capacity <= 0:
        parser.error(f"Argument --capacity, if passed, must be > 0 (got {args.capacity})")

    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error(f"Argument --max-tokens, if passed, must be > 0 (got {args.max_tokens})")

    if args.max_tracepoints is not None and args.max_tracepoints <= 0:
        parser.error(
            f"Argument --max-tracepoints, if passed, must be > 0 (got {args.max_tracepoints})"
        )

    if args.threads is not None and args.threads <= 0:
        parser.error(f"Argument --threads, if passed, must be >= 0 (got {args.threads})")

    if args.start_idx is not None and args.start_idx < 0:
        parser.error(f"Argument --start_idx, if passed, must be >= 0 (got {args.start_idx})")

    if args.merge:
        if len(args.merge) != len(set(args.merge)):
            parser.error("Argument --merge cannot contain duplicated folder names")

    return args


def main():
    parser = get_parser()
    args = validate_parser(parser)

    root = Path(args.root)
    vocab = LatexVocab.load(args.vocab)
    out_dir = root if args.out_dir is None else Path(args.out_dir)

    logger.info(f"Preprocessing splits: {', '.join(args.splits)}")
    for split in args.splits:
        preprocess_split(
            root=root,
            out_dir=out_dir,
            split_name=split,
            vocab=vocab,
            num_workers=args.threads,
            start_idx=args.start_idx,
            capacity=args.capacity,
            max_tokens=args.max_tokens,
            max_tracepoints=args.max_tracepoints,
        )

    if to_merge := list(args.merge):
        merge_tmp = Path(out_dir, ".merge_tmp")
        merge_tmp.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(f"Preprocessing merge folders: {', '.join(to_merge)}")
            for merging in to_merge:
                preprocess_split(
                    root=root,
                    out_dir=merge_tmp,
                    split_name=merging,
                    vocab=vocab,
                    num_workers=args.threads,
                    start_idx=args.start_idx,
                    capacity=args.capacity,
                    max_tokens=args.max_tokens,
                    max_tracepoints=args.max_tracepoints,
                )
                merge_pt = merge_tmp / (merging + ".pt")
                if not merge_pt.exists():
                    logger.warning(f"Skipping merge {merging}: no preprocessed file produced")
                    continue

                merge_data = torch.load(merge_pt, weights_only=True)
                merge_into(merge_data=merge_data, out_dir=out_dir, splits=args.splits)

        finally:
            if merge_tmp.exists():
                shutil.rmtree(merge_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
