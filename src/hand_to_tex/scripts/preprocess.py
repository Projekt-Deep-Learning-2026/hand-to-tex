import argparse
import concurrent.futures
import itertools
import os
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from hand_to_tex.datasets import HMEDataset, InkData
from hand_to_tex.utils import LatexVocab

BATCH_SIZE = 10000
SPLITS = ["train", "valid", "test", "symbols"]
DATASET_PATH = Path("./data/mathwriting-2024")
VOCAB_PATH = Path("./data/assets/vocab.json")


def _process_single_file(
    pth: Path, vocab: LatexVocab
) -> (
    tuple[Literal["success"], tuple[torch.Tensor, torch.Tensor]]
    | tuple[Literal["error", "empty"], None]
):
    """Process single .inkml file, return status and pair (features, tokens)"""
    try:
        ink = InkData.load(pth)
        fts = HMEDataset.extract_features(ink)

        if fts.size(0) == 0:
            return "empty", None

        tokens = vocab.encode_expr(ink.tex_norm)
        tokens = torch.tensor(tokens, dtype=torch.long)

        return "success", (fts, tokens)

    except Exception as _e:
        return "error", None


def preprocess_split(
    root: Path,
    out_dir: Path,
    split_name: str,
    vocab: LatexVocab,
    num_workers: int,
    start_idx: int,
    capacity: int | None,
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

    if capacity:
        inkmls = inkmls[:capacity]

    print(
        f"Processing split {split_name}, {len(inkmls)} .inkml files found, using {num_workers} workers"
    )

    temp_files = []
    processed = []
    empty_cnt, err_cnt = 0, 0
    try:
        with (
            concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor,
            tqdm(total=len(inkmls), desc=split_name) as pbar,
        ):
            for b_idx, batch_inkmls in enumerate(itertools.batched(inkmls, BATCH_SIZE)):
                futures = {
                    executor.submit(_process_single_file, pth, vocab): pth for pth in batch_inkmls
                }
                for future in concurrent.futures.as_completed(futures):
                    status, res = future.result()
                    match status:
                        case "success":
                            fts, ts = res[0].clone(), res[1].clone()  # type: ignore
                            processed.append((fts, ts))
                        case "empty":
                            empty_cnt += 1
                        case "error":
                            err_cnt += 1
                    pbar.update(1)

                temp_path = out_dir / f"{split_name}_temp_{b_idx}.pt"
                torch.save(processed, temp_path)
                temp_files.append(temp_path)
                processed = []

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


def main():
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
        "--out_dir",
        type=str,
        required=False,
        help="Directory where .pt preprocessed splits are going to be saved, by default the `root` directory will be used",
    )
    parser.add_argument(
        "--start_idx",
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
    args = parser.parse_args()

    if args.capacity is not None and args.capacity <= 0:
        parser.error("Argument --capacity, when passed, must be > 0")

    root = Path(args.root)
    vocab = LatexVocab.load(args.vocab)
    out_dir = root if args.out_dir is None else Path(args.out_dir)

    print(f"Preprocessing splits: {', '.join(args.splits)}")
    for split in args.splits:
        preprocess_split(root, out_dir, split, vocab, args.threads, args.start_idx, args.capacity)


if __name__ == "__main__":
    main()
