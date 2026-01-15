import argparse
import asyncio
import copy
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tqdm import tqdm

random.seed(42)


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


TRAJECTORY_SUMMARY_TEMPLATE = """You are analyzing a model's response to a memory-based question-answering task.

Your goal is to summarize the reasoning trajectory and identify the key factors that led to a correct or incorrect answer.

**Task Context:**

<question>
{question}
</question>

<retrieved_memories>
{memories}
</retrieved_memories>

<ground_truth_answer>
{answer}
</ground_truth_answer>

<model_response>
{response}
</model_response>

<evaluation>
Correct: {grade}
</evaluation>

**Analysis Instructions:**

1. **Memory Utilization Analysis:**
   - Which memories did the model use? Which did it ignore?
   - Was the memory selection appropriate for answering the question?
   - Did the model correctly interpret the information in the memories?

2. **Reasoning Process:**
   - What reasoning steps did the model take?
   - Were there any logical errors or correct inferences?
   - How did the model synthesize information from multiple memories?

3. **Key Decision Points:**
   - What were the critical decisions that led to success or failure?
   - If correct: What memory utilization strategy worked well?
   - If incorrect: Where did the reasoning go wrong? What information was missed or misinterpreted?

4. **Lesson Learned:**
   - What generalizable insight about memory utilization can be extracted from this case?

Provide a concise summary focusing on the above points:"""


META_MEMORY_UPDATE_TEMPLATE = """You are a meta-learning specialist.
Your task is to analyze multiple attempts at answering the same question and derive generalizable meta-memory principles.
Meta-memories are high-level strategies that teach a model "learning to learn".
Specifically, how to effectively utilize retrieved memory fragments to answer questions.

Question:
{question}

Retrieved Memories:
{memories}

Ground Truth Answer:
{answer}

Summaries of Multiple Attempts:
{summaries}

Current Meta-Memory Knowledge Base:
{meta_memories}

**Your Task:**

1. **Cross-Attempt Analysis:**
   - Compare the successful vs unsuccessful attempts
   - Identify patterns: What strategies led to correct answers?
   - Identify anti-patterns: What mistakes led to incorrect answers?

2. **Derive Meta-Memory Principles:**
   Based on your analysis, propose updates to the meta-memory knowledge base.
   
   Each meta-memory should be:
   - A generalizable strategy about HOW to utilize memories (not domain-specific facts)
   - Actionable guidance that can be applied to future questions
   - Concise (one sentence, max 30 words)
   
   Examples of good meta-memories:
   - "When memories contain temporal information, prioritize the most recent data unless the question asks about history."
   - "Cross-validate facts that appear in multiple memories; single-source claims require more caution."
   - "If memories seem contradictory, check if they refer to different time periods or contexts."

3. **Propose Operations:**
   - **add**: Add a new meta-memory (when you discover a new principle not covered by existing ones)
   - **update**: Update an existing meta-memory by ID (when you can improve or refine an existing principle)
   - **delete**: Delete an existing meta-memory by ID (when a principle is wrong or redundant)

**Output Format:**

First, provide your reasoning. Then output a JSON array:

```json
[
    {{"operation": "add", "content": "Your new meta-memory principle"}},
    {{"operation": "update", "id": "M0", "content": "Updated meta-memory principle"}},
    {{"operation": "delete", "id": "M1"}}
]
```

Note: Quality over quantity - only propose operations that genuinely improve the meta-memory knowledge base."""


BATCH_META_MEMORY_UPDATE_TEMPLATE = """You are a meta-learning specialist responsible for consolidating meta-memory updates from multiple learning samples.

A batch of questions has been analyzed, and each analysis proposed some operations to update the meta-memory knowledge base. Your task is to consolidate these proposals, resolve conflicts, and produce the final update plan.

**Current Meta-Memory Knowledge Base:**
{existing_meta_memories}

**Proposed Operations from This Batch:**
(Each operation was proposed after analyzing a specific question)
{proposed_updates}

**Consolidation Guidelines:**

1. **Merge Similar Proposals:**
   - If multiple proposals suggest similar principles, merge them into a single, more general formulation
   - Prefer broader applicability over narrow specificity

2. **Resolve Conflicts:**
   - If proposals contradict each other, analyze which is more generally valid
   - Consider how many different questions support each proposal
   - When in doubt, prefer the more cautious/conservative principle

3. **Avoid Redundancy:**
   - Do not add a new meta-memory if it overlaps significantly with an existing one
   - Instead, update the existing one to incorporate the new insight

4. **Quality Control:**
   - Each final meta-memory must be about memory utilization strategy
   - Each must be generalizable (not specific to one question type)
   - Each must be actionable and concise (max 30 words)

5. **Maintain Stability:**
   - Don't delete meta-memories unless they are clearly wrong or completely redundant
   - Prefer updating over deleting + adding

**Output Format:**

First, analyze the proposed operations and explain your consolidation decisions.

Then output the final operations to apply:

```json
[
    {{"operation": "add", "content": "Consolidated new meta-memory"}},
    {{"operation": "update", "id": "M0", "content": "Consolidated updated meta-memory"}},
    {{"operation": "delete", "id": "M2"}}
]
```

Note: The number of final operations should typically be less than the number of proposals due to consolidation."""


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


class MetaMemoryUpdater:
    def __init__(
        self,
        model_name="qwen3-30b",
        base_url="http://localhost:29001/v1",
        api_key="xxx",
    ):
        self.llm = LLM(model_name=model_name, base_url=base_url, api_key=api_key)

    def run(
        self,
        rollouts: list[dict],
        meta_memories: dict,
        save_dir: str,
        max_workers: int = 8,
        only_partial_correct: bool = True,
    ) -> dict:
        problem_to_rollouts = defaultdict(list)
        for r in rollouts:
            if r.get("response"):
                problem_to_rollouts[r["problem"]].append(r)

        filtered_problems = {}
        for problem, group in problem_to_rollouts.items():
            if only_partial_correct:
                scores = [r.get("reward", 0) for r in group]
                avg = sum(scores) / len(scores) if scores else 0
                if 0 < avg < 1:
                    filtered_problems[problem] = group
            else:
                filtered_problems[problem] = group

        print(f"Processing {len(filtered_problems)} problems with partial correctness")

        if not filtered_problems:
            print("No problems with partial correctness, skipping update")
            return copy.deepcopy(meta_memories)

        problem_to_summaries = self._summarize_trajectories(
            filtered_problems, save_dir, max_workers
        )

        all_operations = self._extract_updates(
            problem_to_summaries, meta_memories, save_dir, max_workers
        )

        new_meta_memories = self._consolidate_and_apply(
            meta_memories, all_operations, save_dir
        )

        new_meta_memories = {
            f"M{i}": v for i, v in enumerate(new_meta_memories.values())
        }
        return new_meta_memories

    def _summarize_trajectories(
        self,
        problem_groups: dict,
        save_dir: str,
        max_workers: int,
    ) -> dict:
        cache_file = os.path.join(save_dir, "trajectory_summaries.json")
        if os.path.exists(cache_file):
            print("Loading cached trajectory summaries")
            return json.load(open(cache_file))

        all_rollouts = []
        for problem, group in problem_groups.items():
            for r in group:
                all_rollouts.append(r)

        def summarize_one(rollout):
            try:
                prompt = TRAJECTORY_SUMMARY_TEMPLATE.format(
                    question=rollout["problem"],
                    memories=rollout.get("memories", "N/A"),
                    answer=rollout.get("groundtruth", "N/A"),
                    response=rollout.get("response", "N/A"),
                    grade="CORRECT" if rollout.get("reward", 0) > 0 else "INCORRECT",
                )
                summary = self.llm.chat(prompt)
                return {
                    "problem": rollout["problem"],
                    "response": rollout.get("response", ""),
                    "reward": rollout.get("reward", 0),
                    "summary": summary,
                }
            except Exception as e:
                print(f"Failed to summarize trajectory: {e}")
                return None

        summaries = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(summarize_one, r): r for r in all_rollouts}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Summarizing trajectories",
            ):
                result = future.result()
                if result:
                    summaries.append(result)

        problem_to_summaries = defaultdict(list)
        for s in summaries:
            problem_to_summaries[s["problem"]].append(s)

        with open(cache_file, "w") as f:
            json.dump(dict(problem_to_summaries), f, indent=2, ensure_ascii=False)

        return dict(problem_to_summaries)

    def _extract_updates(
        self,
        problem_to_summaries: dict,
        meta_memories: dict,
        save_dir: str,
        max_workers: int,
    ) -> list[dict]:
        cache_file = os.path.join(save_dir, "proposed_updates.json")
        if os.path.exists(cache_file):
            print("Loading cached proposed updates")
            return json.load(open(cache_file))

        def process_problem(problem, summaries):
            try:
                summaries_str = "\n\n".join(
                    [
                        f"--- Attempt {i+1} ({'CORRECT' if s['reward'] > 0 else 'INCORRECT'}) ---\n{s['summary']}"
                        for i, s in enumerate(summaries)
                    ]
                )

                first_summary = summaries[0] if summaries else {}

                meta_str = (
                    "\n".join([f"[{k}] {v}" for k, v in meta_memories.items()])
                    if meta_memories
                    else "None"
                )

                prompt = META_MEMORY_UPDATE_TEMPLATE.format(
                    question=problem,
                    memories=first_summary.get("memories", "N/A"),
                    answer=first_summary.get("groundtruth", "N/A"),
                    summaries=summaries_str,
                    meta_memories=meta_str,
                )

                response = self.llm.chat(prompt)
                json_str = response.split("```json")[-1].split("```")[0].strip()
                operations = json.loads(json_str)

                return {
                    "problem": problem,
                    "operations": operations,
                    "reasoning": response,
                }
            except Exception as e:
                print(f"Failed to extract updates for problem: {e}")
                return None

        rollout_file = os.path.join(save_dir, "rollout.jsonl")
        problem_to_data = {}
        if os.path.exists(rollout_file):
            with open(rollout_file) as f:
                for line in f:
                    r = json.loads(line)
                    if r["problem"] not in problem_to_data:
                        problem_to_data[r["problem"]] = r

        for problem, summaries in problem_to_summaries.items():
            if problem in problem_to_data:
                orig = problem_to_data[problem]
                for s in summaries:
                    s["memories"] = orig.get("memories", "N/A")
                    s["groundtruth"] = orig.get("groundtruth", "N/A")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_problem, p, s): p
                for p, s in problem_to_summaries.items()
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Extracting updates"
            ):
                result = future.result()
                if result:
                    results.append(result)

        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def _consolidate_and_apply(
        self,
        meta_memories: dict,
        proposed_updates: list[dict],
        save_dir: str,
    ) -> dict:
        cache_file = os.path.join(save_dir, "applied_updates.json")
        if os.path.exists(cache_file):
            print("Loading cached applied updates")
            return json.load(open(cache_file)).get("new_meta_memories", {})

        all_ops = []
        for update in proposed_updates:
            for op in update.get("operations", []):
                op_with_source = copy.deepcopy(op)
                op_with_source["source_problem"] = update.get("problem", "unknown")
                all_ops.append(op_with_source)

        print(f"Total proposed operations: {len(all_ops)}")

        if not all_ops:
            return copy.deepcopy(meta_memories)

        new_memories = self._consolidate_with_llm(meta_memories, all_ops)

        with open(cache_file, "w") as f:
            json.dump(
                {
                    "all_operations": all_ops,
                    "new_meta_memories": new_memories,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return new_memories

    def _consolidate_with_llm(
        self, meta_memories: dict, operations: list[dict]
    ) -> dict:
        existing_str = (
            "\n".join([f"[{k}] {v}" for k, v in meta_memories.items()])
            if meta_memories
            else "None"
        )

        ops_formatted = []
        for op in operations:
            op_str = f"From question analysis: {op.get('source_problem', 'unknown')[:50]}...\n"
            op_str += f"  Operation: {op.get('operation', 'unknown')}"
            if op.get("id"):
                op_str += f", ID: {op['id']}"
            if op.get("content"):
                op_str += f"\n  Content: {op['content']}"
            ops_formatted.append(op_str)

        ops_str = "\n\n".join(ops_formatted)

        prompt = BATCH_META_MEMORY_UPDATE_TEMPLATE.format(
            existing_meta_memories=existing_str,
            proposed_updates=ops_str,
        )

        try:
            response = self.llm.chat(prompt)
            json_str = response.split("```json")[-1].split("```")[0].strip()
            consolidated_ops = json.loads(json_str)
            return self._direct_apply(meta_memories, consolidated_ops)
        except Exception as e:
            print(f"Consolidation failed: {e}")
            return self._direct_apply(meta_memories, operations)

    def _direct_apply(self, meta_memories: dict, operations: list[dict]) -> dict:
        new_memories = copy.deepcopy(meta_memories)
        next_id = len(new_memories)

        for op in operations:
            try:
                op_type = op.get("operation", "").lower()

                if op_type == "add":
                    content = op.get("content", "")
                    if content:
                        new_memories[f"M{next_id}"] = content
                        next_id += 1

                elif op_type == "update":
                    target_id = op.get("id", "")
                    content = op.get("content", "")
                    if target_id in new_memories and content:
                        new_memories[target_id] = content

                elif op_type == "delete":
                    target_id = op.get("id", "")
                    if target_id in new_memories:
                        del new_memories[target_id]

            except Exception as e:
                print(f"Failed to apply operation {op}: {e}")

        return new_memories


async def evaluate(
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


def parse_args():
    parser = argparse.ArgumentParser(description="Training-Free Meta Memory Learning")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/folds_5_split/fold_1_train.json",
    )
    parser.add_argument("--dataset_truncate", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples per question"
    )
    parser.add_argument("--rollout_concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--api_key", type=str, default="xxx")
    parser.add_argument("--base_url", type=str, default="http://localhost:29001/v1")
    parser.add_argument("--model_name", type=str, default="qwen3-30b")
    parser.add_argument(
        "--judge_base_url", type=str, default="http://localhost:29002/v1"
    )
    parser.add_argument("--judge_model_name", type=str, default="qwen-235b")
    return parser.parse_args()


async def main():
    args = parse_args()

    global judge_llm
    judge_llm = LLM(
        model_name=args.judge_model_name,
        base_url=args.judge_base_url,
        api_key=args.api_key,
    )

    experiment_dir = os.path.join("data", "memory", "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    train_data = load_data(args.dataset)
    print(f"Loaded {len(train_data)} records")

    if args.dataset_truncate:
        train_data = train_data[: args.dataset_truncate]
        print(f"Truncated to {len(train_data)} records")

    if len(train_data) % args.batchsize != 0:
        new_size = (len(train_data) // args.batchsize) * args.batchsize
        train_data = train_data[:new_size]
        print(f"Adjusted to {len(train_data)} records for batch size {args.batchsize}")

    stats_file = os.path.join(experiment_dir, "stats.json")
    stats = json.load(open(stats_file)) if os.path.exists(stats_file) else {}

    for epoch in range(args.epochs):
        print("=" * 50)
        print(f"Epoch {epoch}")
        print("=" * 50)

        epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        shuffled_file = os.path.join(epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_file):
            shuffled_data = []
            with open(shuffled_file) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
        else:
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_file, "w") as f:
                for item in shuffled_data:
                    f.write(json.dumps(item) + "\n")

        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            step_key = f"step_{step}"

            if step_key in stats and stats[step_key].get("complete"):
                print(f"Step {step} already complete, skipping")
                continue

            print(f"\nStep {step} (Epoch {epoch}, Batch {batch_idx})")
            step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)

            batch_start = batch_idx * args.batchsize
            batch_end = (batch_idx + 1) * args.batchsize
            batch_data = copy.deepcopy(shuffled_data[batch_start:batch_end])

            if step > 0:
                prev_mm_file = os.path.join(
                    experiment_dir, f"step_{step}", "meta_memories.json"
                )
                if os.path.exists(prev_mm_file):
                    meta_memories = json.load(open(prev_mm_file))
                else:
                    meta_memories = {}
            else:
                meta_memories = {}

            formatted_mm = (
                "\n".join([f"[{k}] {v}" for k, v in meta_memories.items()])
                if meta_memories
                else "None"
            )

            formatted_batch = []
            for item in batch_data:
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

Think step by step and provide your answer."""

                formatted_batch.append({"prompt": prompt, **item})

            formatted_batch = formatted_batch * args.num_samples
            print(
                f"Sampling number: {args.num_samples}, total samples: {len(formatted_batch)}"
            )

            rollout_file = os.path.join(step_dir, "rollout.jsonl")
            rollouts, rollout_stats = await rollout_dataset(
                data=formatted_batch,
                rollout_filename=rollout_file,
                rollout_concurrency=args.rollout_concurrency,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key,
            )

            stats[step_key] = {
                "epoch": epoch,
                "batch": batch_idx,
                "rollout": rollout_stats,
            }

            next_step_dir = os.path.join(experiment_dir, f"step_{step + 1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_mm_file = os.path.join(next_step_dir, "meta_memories.json")

            if not os.path.exists(next_mm_file):
                updater = MetaMemoryUpdater(
                    model_name=args.model_name,
                    base_url=args.base_url,
                    api_key=args.api_key,
                )
                new_meta_memories = updater.run(
                    rollouts=rollouts,
                    meta_memories=meta_memories,
                    save_dir=step_dir,
                    max_workers=args.rollout_concurrency,
                    only_partial_correct=args.num_samples > 1,
                )
                json.dump(new_meta_memories, open(next_mm_file, "w"), indent=2)
                print(f"Saved {len(new_meta_memories)} meta-memories")

            stats[step_key]["complete"] = True
            json.dump(stats, open(stats_file, "w"), indent=2)

    print("\nTraining complete!")

    final_step = args.epochs * num_batches
    final_mm_file = os.path.join(
        experiment_dir, f"step_{final_step}", "meta_memories.json"
    )
    final_mm = {}
    if os.path.exists(final_mm_file):
        final_mm = json.load(open(final_mm_file))
        print(f"\nFinal Meta-Memories ({len(final_mm)} entries):")
        for k, v in final_mm.items():
            print(f"  [{k}] {v}")


if __name__ == "__main__":
    asyncio.run(main())
