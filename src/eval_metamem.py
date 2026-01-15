import argparse
import asyncio
import copy
import json
import os
import re
import time
from collections import defaultdict

import openai
from tqdm import tqdm


class LLM:
    def __init__(
        self,
        model_name="qwen3-30b",
        base_url="http://localhost:29001/v1",
        api_key="xxx",
    ):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages_or_prompt, max_tokens=16384, temperature=0, max_retries=3):
        for attempt in range(max_retries):
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError(
                        "messages_or_prompt must be a string or a list of messages."
                    )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"LLM error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
        return ""


PROBLEM_WITH_META_MEMORY_TEMPLATE = """You are an intelligent assistant with access to a memory system.
Your task is to answer the user's question by effectively utilizing the retrieved memory fragments.

**Meta-Memory Guidelines (Learning to Learn):**
The following are learned strategies that teach you how to effectively utilize memory fragments. Apply these guidelines when processing the retrieved memories:
{meta_memories}

**User Question:**
{question}

**Retrieved Memory Fragments:**
{memories}

**Instructions:**
1. First, review the meta-memory guidelines to understand how to approach memory utilization
2. Analyze the retrieved memory fragments and identify relevant information
3. Apply the meta-memory strategies to synthesize information from memories
4. Formulate your answer based on the synthesized knowledge
5. If memories are insufficient or conflicting, handle according to the guidelines

Think step by step about how to utilize the memories, then provide your final answer."""


def load_data(filepath: str) -> list[dict]:
    """
    Load dataset from file.

    Expected format per sample:
    {
        "question_id": "unique_id",  # optional, will auto-generate if missing
        "problem": "User's question",
        "memories": ["memory1", "memory2", ...] or "formatted memories string",
        "groundtruth": "expected answer"
    }
    """
    data = []

    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("data", [data])

    normalized = []
    for item in data:
        problem = item.get("problem") or item.get("question")
        if not problem:
            continue

        question_id = item.get("question_id")

        memories = item.get("memories") or item.get("related_memories")

        groundtruth = (
            item.get("groundtruth")
            or item.get("answer")
            or item.get("ground_truth", "")
        )

        task = item.get("task")

        normalized.append(
            {
                "question_id": str(question_id),
                "problem": problem,
                "memories": memories,
                "groundtruth": str(groundtruth),
                "task": task,
            }
        )

    return normalized


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def true_or_false(response):
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = (
        first_line.replace(".", "")
        .replace("!", "")
        .replace(":", "")
        .replace(";", "")
        .split()
    )
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


judge_llm = LLM(model_name="qwen3-235b", base_url="http://localhost:29002/v1")


def verify_answer(sample: dict, ground_truth: str) -> float:
    """Verify if the model's response matches the ground truth."""
    response = sample.get("response", "")
    is_abs = True if "abs" in sample["question_id"] else False

    judge_prompt = get_anscheck_prompt(
        sample["task"],
        sample["problem"],
        ground_truth,
        response,
        is_abs,
    )
    response = judge_llm.chat(judge_prompt)
    return true_or_false(response)


async def rollout_dataset(
    data: list[dict],
    rollout_filename: str,
    rollout_concurrency: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model_name: str = "qwen3-30b",
    base_url: str = "http://localhost:29001/v1",
    api_key: str = "xxx",
) -> tuple[list[dict], dict]:
    """Rollout the dataset using LLM prompts."""

    rollouts = []
    if os.path.exists(rollout_filename):
        with open(rollout_filename, "r", encoding="utf-8") as f:
            for line in f:
                rollouts.append(json.loads(line))

    if len(rollouts) > 0:
        data_problems = [each["problem"] for each in data]
        rollouts_problems = [each["problem"] for each in rollouts]
        assert (
            data_problems == rollouts_problems
        ), "Data mismatch with existing rollouts"
    else:
        rollouts = [{"runid": i, **sample} for i, sample in enumerate(data)]

    with open(rollout_filename, "w", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    task_queue = asyncio.Queue()
    pending_count = 0
    for sample in rollouts:
        if "response" not in sample or not sample["response"]:
            sample["retry_count"] = 0
            await task_queue.put(copy.deepcopy(sample))
            pending_count += 1

    pbar = tqdm(total=pending_count, desc="Rolling out")

    async def worker(worker_id: int):
        llm = LLM(model_name=model_name, base_url=base_url, api_key=api_key)
        while not task_queue.empty():
            sample = await task_queue.get()
            try:
                prompt = sample["prompt"]
                response = await asyncio.to_thread(
                    llm.chat, prompt, temperature=temperature, max_tokens=max_tokens
                )

                sample["response"] = response
                sample["reward"] = verify_answer(sample, sample["groundtruth"])

                rollouts[sample["runid"]] = sample
                with open(rollout_filename, "w", encoding="utf-8") as f:
                    for r in rollouts:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                pbar.update(1)

            except Exception as e:
                sample["retry_count"] = sample.get("retry_count", 0) + 1
                if sample["retry_count"] <= 3:
                    await task_queue.put(sample)
                else:
                    sample["response"] = f"Error: {e}"
                    sample["reward"] = 0
                    rollouts[sample["runid"]] = sample
                    pbar.update(1)
            finally:
                task_queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(rollout_concurrency)]
    await task_queue.join()
    for w in workers:
        w.cancel()
    pbar.close()

    all_rewards = [r.get("reward", 0) for r in rollouts]
    problem_to_scores = defaultdict(list)
    for r in rollouts:
        problem_to_scores[r["problem"]].append(r.get("reward", 0))

    stats = {
        "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        "num_samples": len(rollouts),
    }
    print(f"Stats: {stats}")

    return rollouts, stats


async def evaluate_single_step(
    test_data: list[dict],
    meta_memories: dict,
    eval_dir: str,
    rollout_concurrency: int = 8,
    max_tokens: int = 4096,
    model_name: str = "qwen3-30b",
    base_url: str = "http://localhost:29001/v1",
    api_key: str = "xxx",
) -> dict:
    os.makedirs(eval_dir, exist_ok=True)

    formatted_mm = (
        "\n".join([f"[{k}] {v}" for k, v in meta_memories.items()])
        if meta_memories
        else "None"
    )

    formatted_data = []
    for item in test_data:
        if meta_memories:
            prompt = PROBLEM_WITH_META_MEMORY_TEMPLATE.format(
                meta_memories=formatted_mm,
                question=item["problem"],
                memories=item.get("memories", "No memories available"),
            )
        else:
            prompt = f"""Answer the following question based on the retrieved memory fragments.

**Question:**
{item["problem"]}

**Retrieved Memory Fragments:**
{item.get("memories", "No memories available")}

Provide your final answer."""

        formatted_data.append({"prompt": prompt, **item})

    rollout_file = os.path.join(eval_dir, "eval_rollout.jsonl")
    rollouts, _ = await rollout_dataset(
        data=formatted_data,
        rollout_filename=rollout_file,
        rollout_concurrency=rollout_concurrency,
        temperature=0,
        max_tokens=max_tokens,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )

    total = len(rollouts)
    correct = sum(1 for r in rollouts if r.get("reward", 0) > 0)
    accuracy = correct / total if total > 0 else 0

    task_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rollouts:
        task = r.get("task", "unknown")
        task_results[task]["total"] += 1
        if r.get("reward", 0) > 0:
            task_results[task]["correct"] += 1

    task_accuracy = {
        task: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for task, stats in task_results.items()
    }

    results = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "task_accuracy": task_accuracy,
        "num_meta_memories": len(meta_memories),
    }

    results_file = os.path.join(eval_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print("Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Meta-memories used: {len(meta_memories)}")
    print("\nAccuracy by task type:")
    for task, acc in sorted(task_accuracy.items()):
        task_stats = task_results[task]
        print(f"  {task}: {acc:.4f} ({task_stats['correct']}/{task_stats['total']})")
    print(f"{'=' * 50}\n")

    return results


def find_step_folders(
    experiment_dir: str, start_step: int = 1
) -> list[tuple[int, str]]:
    step_folders = []
    pattern = re.compile(r"^step_(\d+)$")

    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")

    for name in os.listdir(experiment_dir):
        match = pattern.match(name)
        if match:
            step_num = int(match.group(1))
            if step_num >= start_step:
                folder_path = os.path.join(experiment_dir, name)
                meta_file = os.path.join(folder_path, "meta_memories.json")
                if os.path.exists(meta_file):
                    step_folders.append((step_num, folder_path))

    step_folders.sort(key=lambda x: x[0])
    return step_folders


async def evaluate_all_steps(
    experiment_dir: str,
    test_data: list[dict],
    output_file: str,
    output_dir: str,
    rollout_concurrency: int = 8,
    max_tokens: int = 4096,
    start_step: int = 1,
    include_baseline: bool = True,
    model_name: str = "qwen3-30b",
    base_url: str = "http://localhost:29001/v1",
    api_key: str = "xxx",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    step_folders = find_step_folders(experiment_dir, start_step)
    print(f"Found {len(step_folders)} step folders to evaluate")

    if not step_folders:
        print("No step folders found!")
        return {}

    all_results = {
        "experiment_dir": experiment_dir,
        "test_dataset_size": len(test_data),
        "steps": {},
    }

    if include_baseline:
        print("\n" + "=" * 60)
        print("Evaluating Baseline (No Meta-Memories)")
        print("=" * 60)

        baseline_dir = os.path.join(output_dir, "baseline")
        baseline_results = await evaluate_single_step(
            test_data=test_data,
            meta_memories={},
            eval_dir=baseline_dir,
            rollout_concurrency=rollout_concurrency,
            max_tokens=max_tokens,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        all_results["baseline"] = baseline_results

    for step_num, folder_path in step_folders:
        print("\n" + "=" * 60)
        print(f"Evaluating Step {step_num}")
        print("=" * 60)

        meta_file = os.path.join(folder_path, "meta_memories.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_memories = json.load(f)

        print(f"Loaded {len(meta_memories)} meta-memories from {meta_file}")
        for k, v in meta_memories.items():
            print(f"  [{k}] {v[:80]}..." if len(v) > 80 else f"  [{k}] {v}")

        step_eval_dir = os.path.join(output_dir, f"step_{step_num}")

        step_results = await evaluate_single_step(
            test_data=test_data,
            meta_memories=meta_memories,
            eval_dir=step_eval_dir,
            rollout_concurrency=rollout_concurrency,
            max_tokens=max_tokens,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )

        all_results["steps"][f"step_{step_num}"] = {
            "step_num": step_num,
            "meta_memories_path": meta_file,
            "meta_memories": meta_memories,
            "results": step_results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if include_baseline and "baseline" in all_results:
        print(f"Baseline (no meta-memories): {all_results['baseline']['accuracy']:.4f}")

    print("\nStep-wise accuracy:")
    for step_key in sorted(
        all_results["steps"].keys(), key=lambda x: int(x.split("_")[1])
    ):
        step_info = all_results["steps"][step_key]
        acc = step_info["results"]["accuracy"]
        num_mm = step_info["results"]["num_meta_memories"]
        print(f"  {step_key}: {acc:.4f} ({acc * 100:.2f}%) - {num_mm} meta-memories")

    if all_results["steps"]:
        best_step = max(
            all_results["steps"].items(), key=lambda x: x[1]["results"]["accuracy"]
        )
        print(
            f"\nBest performing step: {best_step[0]} with accuracy {best_step[1]['results']['accuracy']:.4f}"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return all_results


async def main(args):

    global judge_llm
    judge_llm = LLM(
        model_name=args.judge_model_name,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
    )

    test_data = load_data(args.dataset)
    print(f"Loaded {len(test_data)} test samples from {args.dataset}")

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(args.output_dir, "eval_all_steps.json")

    results = await evaluate_all_steps(
        experiment_dir=args.experiment_dir,
        test_data=test_data,
        output_file=output_file,
        output_dir=args.output_dir,
        rollout_concurrency=args.rollout_concurrency,
        max_tokens=args.max_tokens,
        start_step=args.start_step,
        include_baseline=args.include_baseline,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate meta-memories from each training step"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing step folders (e.g., data/memory/train/fold_1_train)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the test dataset file (jsonl or json format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results for each step",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the aggregated results JSON (default: {output_dir}/eval_all_steps.json)",
    )
    parser.add_argument(
        "--rollout_concurrency",
        type=int,
        default=8,
        help="Number of parallel workers for rollout",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens per response",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=1,
        help="Start evaluating from which step (default: 1, skip step_0)",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        default=True,
        help="Include baseline (no meta-memories) evaluation",
    )
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip baseline evaluation",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="xxx",
        help="API key for generation model",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:29001/v1",
        help="Base URL for generation model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3-30b",
        help="Model name for generation",
    )
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default="xxx",
        help="API key for judge model",
    )
    parser.add_argument(
        "--judge_base_url",
        type=str,
        default="http://localhost:29002/v1",
        help="Base URL for judge model",
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default="qwen3-235b",
        help="Model name for judge model",
    )

    args = parser.parse_args()

    if args.no_baseline:
        args.include_baseline = False

    asyncio.run(main(args))
