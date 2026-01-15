import argparse
import json
import os
import time
import traceback
from tqdm import tqdm
from openai import OpenAI
from lightmem.memory.lightmem import LightMemory


def parse_args():
    parser = argparse.ArgumentParser()

    # ============ API Configuration ============
    parser.add_argument(
        "--api_key", type=str, default="sk-xxx", help="API Key for the LLM"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:29001/v1",
        help="Base URL for the LLM API",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="qwen3-30b",
        help="Model name to use (e.g., qwen3-30b)",
    )

    # ============ Model Paths ============
    parser.add_argument(
        "--llmlingua_path",
        type=str,
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        help="Path or HF ID for the LLMLingua model",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Path or HF ID for the embedding model",
    )

    # ============ Data Configuration ============
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/longmemeval_s_cleaned.json",
        help="Input JSON data path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/longmemeval_lightmem.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--qdrant_dir",
        type=str,
        default="qdrant/lightmem_qwen3_30b",
        help="Directory for Qdrant storage",
    )

    return parser.parse_args()


class LLMModel:
    def __init__(self, model_name, api_key, base_url):
        self.name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = 2000
        self.temperature = 0.0
        self.top_p = 0.8
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(self, messages: list, **kwargs):
        max_retries = kwargs.get("max_retries", 3)

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                )
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise


def load_lightmem(collection_name, args):
    """
    Initializes LightMemory using arguments from argparse
    """
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": args.llmlingua_path,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
            },
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "hybrid",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": args.llm_model,
                "api_key": args.api_key,
                "max_tokens": 16000,
                "openai_base_url": args.base_url,
            },
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": args.embedding_path,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f"{args.qdrant_dir}/{collection_name}",
            },
        },
        "update": "offline",
    }
    lightmem = LightMemory.from_config(config)
    return lightmem


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading data from: {args.data_path}")
    data = json.load(open(args.data_path, "r"))

    INIT_RESULT = {"add_input_prompt": [], "add_output_prompt": [], "api_call_nums": 0}

    res = []

    print(f"Starting processing using model: {args.llm_model}")
    print(f"Qdrant storage path: {args.qdrant_dir}")

    for item in tqdm(data):
        qid = item.get("question_id")

        try:
            lightmem = load_lightmem(collection_name=qid, args=args)

            sessions = item["haystack_sessions"]
            timestamps = item["haystack_dates"]

            results_list = []

            for session, timestamp in zip(sessions, timestamps):

                while session and session[0]["role"] != "user":
                    session.pop(0)

                num_turns = len(session) // 2
                for turn_idx in range(num_turns):
                    turn_messages = session[turn_idx * 2 : turn_idx * 2 + 2]

                    if (
                        len(turn_messages) < 2
                        or turn_messages[0]["role"] != "user"
                        or turn_messages[1]["role"] != "assistant"
                    ):
                        continue

                    for msg in turn_messages:
                        msg["time_stamp"] = timestamp

                    is_last_turn = (session is sessions[-1]) and (
                        turn_idx == num_turns - 1
                    )

                    result = lightmem.add_memory(
                        messages=turn_messages,
                        force_segment=is_last_turn,
                        force_extract=is_last_turn,
                    )

                    if result != INIT_RESULT:
                        results_list.append(result)

            related_memories = lightmem.retrieve(item["question"], limit=20)
            item["related_memories"] = related_memories
            res.append(item)

        except Exception as e:
            print(f"[ERROR] Failed to process item {qid}: {e}")
            traceback.print_exc()
            continue

    print(f"Saving results to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
