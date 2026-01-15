import argparse
import json
import random
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="5-fold Cross Validation Splitter with Train/Val/Test support."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/full_train.json",
        help="Path to the full dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/folds",
        help="Directory to save the split files",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=500,
        help="Total number of samples to use for the split",
    )
    parser.add_argument(
        "--test_size", type=int, default=100, help="Size of Test set per fold"
    )
    parser.add_argument(
        "--val_size", type=int, default=50, help="Size of Validation set per fold"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_available = len(data)
    print(f"Total raw samples available: {total_available}")

    if total_available < args.total_samples:
        print(
            f"[Warning] Data count ({total_available}) is less than requested total_samples ({args.total_samples})."
        )
        print("Using all available data, but splits might be smaller than requested.")

        args.total_samples = total_available

    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    data_for_split = shuffled_data[: args.total_samples]

    train_size = args.total_samples // args.n_folds - args.test_size

    calculated_train_size = len(data_for_split) - args.test_size - args.val_size

    print("=" * 40)
    print(f"Split Configuration (Per Fold):")
    print(f"  - Test Set : {args.test_size}")
    print(f"  - Val Set  : {args.val_size}")
    print(f"  - Train Set: {calculated_train_size} (Calculated from remainder)")
    print(f"  - Total Used: {len(data_for_split)}")
    print("=" * 40)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(args.n_folds):
        test_start = fold_idx * args.test_size
        test_end = test_start + args.test_size

        test_set = data_for_split[test_start:test_end]
        rest_data = data_for_split[:test_start] + data_for_split[test_end:]

        val_set = rest_data[: args.val_size]
        train_set = rest_data[args.val_size :]

        splits = {"train": train_set, "val": val_set, "test": test_set}

        print(f"Processing Fold {fold_idx + 1}...")

        for split_name, split_data in splits.items():
            filename = output_dir / f"fold_{fold_idx + 1}_{split_name}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(split_data, f, ensure_ascii=False, indent=4)

            print(
                f"  -> Saved {split_name:<5}: {len(split_data):3d} items to {filename}"
            )

    print(f"\nSuccessfully finished {args.n_folds}-fold split task!")
    print(f"Results saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
