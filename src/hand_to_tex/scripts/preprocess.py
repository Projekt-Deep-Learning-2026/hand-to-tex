import argparse
import concurrent.futures
import itertools
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from hand_to_tex.datasets import HMEDataset, InkData
from hand_to_tex.utils import LatexVocab

BATCH_SIZE = 10000


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


def preprocess_split(root: Path, split_name: str, vocab: LatexVocab, num_workers: int):
    split_dir = root / split_name
    if not split_dir.exists():
        print(f"Directory {split_dir} not found")
        return

    inkmls = sorted(split_dir.rglob("*.inkml"))
    if not inkmls:
        print(f"No .inkml files found in {split_dir}")
        return

    print(
        f"Processing split {split_name}, {len(inkmls)} .inkml files found using {num_workers} workers"
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

                temp_path = root / f"{split_name}_temp_{b_idx}.pt"
                torch.save(processed, temp_path)
                temp_files.append(temp_path)
                processed = []

        out_file_path = Path(root, split_name + ".pt")
        final_data = []

        print(f"Saving to {out_file_path}")
        for temp in temp_files:
            final_data.extend(torch.load(temp, weights_only=True))
            temp.unlink()

        torch.save(final_data, out_file_path)

        print(f"Preprocessing {split_name}, summary:")
        print(f"Saved: {len(final_data)} out of {len(inkmls)}")
        print(f"Empty samples: {empty_cnt}")
        print(f"Error samples: {err_cnt}")
        print("=" * 50)

    finally:
        for temp in tqdm(temp_files, "Cleaning temp files"):
            temp.unlink()
        print(f"Aborted succesfully, cleaned {len(temp_files)} files")


def main():
    parser = argparse.ArgumentParser(description="Pre-processing of dataset Hand-to-TeX")
    parser.add_argument(
        "--root",
        type=str,
        required=False,
        help="Path to dataset containing all .inkml files (contains train, valid, test), default=data/mathwriting-2024",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=False,
        help="Path to .json file containing vocabulary mapping for latex symbols, default=data/assets/vocab.json",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of workers (processes) to use for preprocessing, default=1",
    )
    args = parser.parse_args()

    root = Path("data/mathwriting-2024" if args.root is None else args.root)
    vocab = LatexVocab.default() if args.vocab is None else LatexVocab.load(args.vocab)

    splits = ["valid", "test"]
    for split in splits:
        preprocess_split(root, split, vocab, args.threads)


if __name__ == "__main__":
    main()
