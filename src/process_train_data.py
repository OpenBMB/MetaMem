import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge raw evaluation data with processed memory results."
    )

    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/longmemeval_s_cleaned.json",
        help="Path to the original raw dataset (contains questions and types)",
    )

    parser.add_argument(
        "--processed_data_path",
        type=str,
        default="output/memory_qwen3_30b.json",
        help="Path to the processed memory data (contains memories)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output/full_train_set.json",
        help="Path to save the final merged JSON",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading raw data from: {args.raw_data_path}")
    if not os.path.exists(args.raw_data_path):
        print(f"Error: Raw data file not found at {args.raw_data_path}")
        sys.exit(1)

    with open(args.raw_data_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    id2item_mp = {}
    for item in ds:
        if "question_id" in item:
            id2item_mp[item["question_id"]] = item

    print(f"Mapped {len(id2item_mp)} items from raw data.")

    print(f"Loading processed memory data from: {args.processed_data_path}")
    if not os.path.exists(args.processed_data_path):
        print(f"Error: Processed data file not found at {args.processed_data_path}")
        sys.exit(1)

    with open(args.processed_data_path, "r", encoding="utf-8") as f:
        processed_data_list = json.load(f)

    ret = []

    for item in processed_data_list:
        question_id = item.get("question_id")

        raw_item = id2item_mp[question_id]

        merged_entry = {
            "question_id": question_id,
            "problem": raw_item.get("question", ""),
            "memories": item.get("related_memories", []),
            "groundtruth": item.get("ground_truth", ""),
            "task": raw_item.get("question_type", ""),
        }
        ret.append(merged_entry)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving {len(ret)} merged items to: {args.output_path}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
