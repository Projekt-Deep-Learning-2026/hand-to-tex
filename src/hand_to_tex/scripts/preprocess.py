import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from hand_to_tex.datasets import HMEDataset, InkData
from hand_to_tex.utils import LatexVocab


def preprocess_split(root: Path, split_name: str, vocab: LatexVocab):

    split_dir = root / split_name
    if not split_dir.exists():
        print(f"Directory {split_dir} not found")
        return

    inkmls = sorted(split_dir.rglob("*.inkml"))
    if not inkmls:
        print(f"No .inkml files found in {split_dir}")
        return

    print(f"Processing split {split_name}, {len(inkmls)} .inkml files found")

    processed = []
    empty_cnt, err_cnt = 0, 0

    for pth in tqdm(inkmls, desc=split_name):
        try:
            ink = InkData.load(pth)
            fts = HMEDataset.extract_features(ink)

            if fts.size(0) == 0:
                empty_cnt += 1
                continue

            tokens = vocab.encode_expr(ink.tex_norm)
            tokens = torch.tensor(tokens, dtype=torch.long)

            processed.append((fts, tokens))

        except Exception as _e:
            err_cnt += 1
            continue

    out_file_path = Path(root, split_name + ".pt")

    print(f"Saving to {out_file_path}")
    torch.save(processed, out_file_path)

    print(f"Preprocessing {split_name}, summary:")
    print(f"Saved: {len(processed)} out of {len(inkmls)}")
    print(f"Empty samples: {empty_cnt}")
    print(f"Error samples: {err_cnt}")
    print("=" * 50)


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
    args = parser.parse_args()

    root = Path("data/mathwriting-2024" if args.root is None else args.root)

    vocab = LatexVocab.default() if args.vocab is None else LatexVocab.load(args.vocab)

    splits = ["train", "valid", "test"]
    for split in splits:
        preprocess_split(root, split, vocab)


if __name__ == "__main__":
    main()
