import argparse
import asyncio
import copy
import json
import os
import time

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


judge_llm = None


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
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
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


def verify_answer(sample: dict, ground_truth: str) -> float:
    """Verify if the model's response matches the ground truth using the Judge LLM."""
    if judge_llm is None:
        return 0.0

    response = sample.get("response", "")
    is_abs = True if "abs" in sample["question_id"] else False

    judge_prompt = get_anscheck_prompt(
        sample["task"],
        sample["problem"],
        ground_truth,
        response,
        is_abs,
    )
    judge_res = judge_llm.chat(judge_prompt)
    return true_or_false(judge_res)


PROBLEM_WITH_META_MEMORY_TEMPLATE = """You are an intelligent assistant with access to a memory system.
Your task is to answer the user's question by effectively utilizing the retrieved memory fragments.

**Meta-Memory Guidelines:**
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


async def inference_dataset(
    data: list[dict],
    meta_memories: dict,
    output_path: str,
    rollout_concurrency: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    model_name: str = "qwen3-30b",
    base_url: str = "http://localhost:29001/v1",
    api_key: str = "xxx",
) -> list[dict]:
    formatted_mm = (
        "\n".join([f"[{k}] {v}" for k, v in meta_memories.items()])
        if meta_memories
        else "None"
    )

    formatted_data = []
    for item in data:
        prompt = PROBLEM_WITH_META_MEMORY_TEMPLATE.format(
            meta_memories=formatted_mm,
            question=item["problem"],
            memories=item.get("memories", "No memories available"),
        )
        formatted_data.append({"prompt": prompt, **item})

    outputs = copy.deepcopy(formatted_data)

    task_queue = asyncio.Queue()
    for sample in outputs:
        await task_queue.put(sample)

    pbar = tqdm(total=len(outputs), desc="Inferencing & Judging")

    async def worker(worker_id: int):
        llm = LLM(model_name=model_name, base_url=base_url, api_key=api_key)
        while not task_queue.empty():
            sample = await task_queue.get()
            try:
                # 1. Generate Response
                response = await asyncio.to_thread(
                    llm.chat,
                    sample["prompt"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                sample["response"] = response

                # 2. Judge Response (Reward)
                # Note: This is done inside the worker, so judge calls are also concurrent
                reward = await asyncio.to_thread(
                    verify_answer, sample, sample["groundtruth"]
                )
                sample["reward"] = 1 if reward else 0

            except Exception as e:
                sample["response"] = f"Error: {e}"
                sample["reward"] = 0
            finally:
                task_queue.task_done()
                pbar.update(1)

    workers = [asyncio.create_task(worker(i)) for i in range(rollout_concurrency)]
    await task_queue.join()
    for w in workers:
        w.cancel()
    pbar.close()

    # Save outputs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Calculate statistics
    total = len(outputs)
    correct = sum(1 for r in outputs if r.get("reward", 0) > 0)
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'=' * 50}")
    print("Inference & Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Output saved to: {output_path}")
    print(f"{'=' * 50}\n")

    return outputs


async def main(args):
    global judge_llm

    judge_llm = LLM(
        model_name=args.judge_model_name,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
    )
    print(f"Judge initialized with model: {args.judge_model_name}")

    data = load_data(args.dataset)
    print(f"Loaded {len(data)} records from {args.dataset}")

    if not os.path.exists(args.meta_memories):
        print(f"Warning: Meta-memories file not found at {args.meta_memories}")
        meta_memories = {}
    else:
        meta_memories = json.load(open(args.meta_memories, "r", encoding="utf-8"))
        print(f"Loaded {len(meta_memories)} meta-memories")

    await inference_dataset(
        data=data,
        meta_memories=meta_memories,
        output_path=args.output,
        rollout_concurrency=args.rollout_concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with meta-memories on a dataset with Judge"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (json/jsonl) with question/memories",
    )
    parser.add_argument(
        "--meta_memories",
        type=str,
        required=True,
        help="Path to meta_memories.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/infer_outputs.jsonl",
        help="Path to save inference results (jsonl)",
    )
    parser.add_argument("--rollout_concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--api_key", type=str, default="xxx")
    parser.add_argument("--base_url", type=str, default="http://localhost:29001/v1")
    parser.add_argument("--model_name", type=str, default="qwen3-30b")
    parser.add_argument("--judge_api_key", type=str, default="xxx")
    parser.add_argument(
        "--judge_base_url", type=str, default="http://localhost:29002/v1"
    )
    parser.add_argument("--judge_model_name", type=str, default="qwen3-235b")

    args = parser.parse_args()
    asyncio.run(main(args))
