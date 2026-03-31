import argparse
import concurrent.futures
import itertools
import os
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from hand_to_tex.datasets import HMEDatasetRaw, InkData
from hand_to_tex.utils import LatexVocab

BATCH_SIZE = 1000
SPLITS = ["train", "valid", "test", "symbols"]
DATASET_PATH = Path("./data/mathwriting-2024")
VOCAB_PATH = Path("./data/assets/vocab.json")


def _process_single_file(
    pth: Path, vocab: LatexVocab
) -> tuple[Literal["success", "error", "empty"], tuple[torch.Tensor, torch.Tensor] | None, str]:
    """Process single .inkml file, return status and pair (features, tokens)"""
    ID = pth.stem
    try:
        ink = InkData.load(pth)
        fts = HMEDatasetRaw.extract_features(ink)

        if fts.size(0) == 0:
            return "empty", None, ID

        tokens = vocab.encode_expr(ink.tex_norm)
        tokens = torch.tensor(tokens, dtype=torch.long)

        return "success", (fts, tokens), ID

    except Exception as _e:
        return "error", None, ID


def _validate(
    elements: dict[str, tuple[torch.Tensor, torch.Tensor]], max_len: None | int, capacity: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:

    if capacity <= 0:
        return []

    res = []
    for _k, (fts, ts) in sorted(elements.items()):
        if max_len and ts.shape[0] > max_len:
            continue
        res.append((fts, ts))

        if len(res) == capacity:
            break

    return res


def preprocess_split(
    root: Path,
    out_dir: Path,
    split_name: str,
    vocab: LatexVocab,
    num_workers: int,
    start_idx: int,
    capacity: int | None,
    max_len: int | None,
):
    split_dir = root / split_name
    if not split_dir.exists():
        print(f"Directory {split_dir} not found")
        return
    if not out_dir.exists():
        print(f"Output directory {out_dir} not found, creating")
        os.mkdir(out_dir)

    inkmls = sorted(split_dir.rglob("*.inkml"))
    if len(inkmls) <= start_idx:
        print(f"No .inkml files found in {split_dir} (after skipping {start_idx} files)")
        return

    inkmls = inkmls[start_idx:]

    if capacity is None:
        capacity = len(inkmls)

    print(
        f"Processing split {split_name}, {len(inkmls)} .inkml files found, using {num_workers} workers"
    )

    total_saved = 0
    temp_files = []
    empty_cnt, err_cnt = 0, 0
    try:
        with (
            concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor,
            tqdm(total=len(inkmls), desc=split_name) as pbar,
        ):
            for b_idx, batch_inkmls in enumerate(itertools.batched(inkmls, BATCH_SIZE)):
                current = {}
                futures = {
                    executor.submit(_process_single_file, pth, vocab): pth for pth in batch_inkmls
                }
                for future in concurrent.futures.as_completed(futures):
                    status, res, ID = future.result()
                    match status:
                        case "success":
                            fts, ts = res[0].clone(), res[1].clone()  # type: ignore
                            current[ID] = (fts, ts)
                        case "empty":
                            empty_cnt += 1
                        case "error":
                            err_cnt += 1
                    pbar.update(1)

                to_save = _validate(current, max_len=max_len, capacity=(capacity - total_saved))

                temp_path = out_dir / f"{split_name}_temp_{b_idx}.pt"
                torch.save(to_save, temp_path)
                temp_files.append(temp_path)

                total_saved += len(to_save)
                pbar.set_postfix(gathered=str(total_saved))

                if len(inkmls) > total_saved >= capacity:
                    pbar.set_description(f"Done, capacity={capacity} achieved")
                    break

        out_file_path = Path(out_dir, split_name + ".pt")
        final_data = []

        print(f"Saving to {out_file_path}")
        for temp in temp_files:
            final_data.extend(torch.load(temp, weights_only=True))

        torch.save(final_data, out_file_path)

        print(f"Preprocessing {split_name}, summary:")
        print(f"Saved: {len(final_data)} out of {len(inkmls)}")
        print(f"Empty samples: {empty_cnt}")
        print(f"Error samples: {err_cnt}")
        print("=" * 50)

    finally:
        for temp in tqdm(temp_files, "Cleaning temp files"):
            temp.unlink()
        print(f"Cleaned {len(temp_files)} temp files succesfully")


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
        "--max-len",
        type=int,
        required=False,
        help="Setting this argument enables you to omit sequences that have more than `max-len` tokens in normalised form",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.capacity is not None and args.capacity <= 0:
        parser.error("Argument --capacity, when passed, must be > 0")

    root = Path(args.root)
    vocab = LatexVocab.load(args.vocab)
    out_dir = root if args.out_dir is None else Path(args.out_dir)

    print(f"Preprocessing splits: {', '.join(args.splits)}")
    for split in args.splits:
        preprocess_split(
            root=root,
            out_dir=out_dir,
            split_name=split,
            vocab=vocab,
            num_workers=args.threads,
            start_idx=args.start_idx,
            capacity=args.capacity,
            max_len=args.max_len,
        )


if __name__ == "__main__":
    main()
