# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the dataset to parquet format
"""

import argparse
import base64
import json
import os
import pickle
import zlib
from functools import partial

from datasets import concatenate_datasets, load_dataset
import pyarrow.parquet as pq

from verl.utils.hdfs_io import copy, makedirs


def to_parquet_compat(dataset, path):
    dataset.to_parquet(path)
    table = pq.read_table(path)
    pq.write_table(table.replace_schema_metadata(None), path)


def example_map_fn(example, idx, process_fn, data_source, ability, split):
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx, "task_id": "", "dataset": data_source},
    }
    return data


def build_aime2024_dataset():
    def process_aime2024(example):
        return example["Problem"], str(example["Answer"])

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_aime2024, data_source=data_source, ability="English", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_gpqa_dimond_dataset():
    import random

    GPQA_QUERY_TEMPLATE = (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
        "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    )

    def process_gpqa_diamond(example):
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    data_source = "Idavidrein/gpqa"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_gpqa_diamond, data_source=data_source, ability="Math", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_cnmo2024_dataset():
    def process_cnmo2024(example):
        return example["question"], example["answer"]

    data_source = "opencompass/LiveMathBench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset_en = load_dataset(data_source, "v202412_CNMO_en", split="test")
    map_fn_en = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_en", ability="Math", split="test"
    )
    dataset_en = dataset_en.map(map_fn_en, with_indices=True, remove_columns=dataset_en.column_names)

    dataset_zh = load_dataset(data_source, "v202412_CNMO_cn", split="test")
    map_fn_zh = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_zh", ability="Math", split="test"
    )
    dataset_zh = dataset_zh.map(map_fn_zh, with_indices=True, remove_columns=dataset_zh.column_names)

    dataset = concatenate_datasets([dataset_en, dataset_zh])
    return dataset


def build_livecodebench_dataset():
    import base64
    import json
    import pickle
    import zlib

    def process_livecodebench(example):
        # Construct Query Prompt
        # From https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py#L140
        query_prompt = (
            f"You will be given a question (problem specification) and will generate a correct Python program "
            f"that matches the specification and passes all tests.\n\nQuestion: {example['question_content']}\n\n"
        )
        if example["starter_code"]:
            query_prompt += (
                f"You will use the following starter code to write the solution to the problem and enclose your "
                f"code within delimiters.\n```python\n{example['starter_code']}\n```"
            )
        else:
            query_prompt += (
                "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
                "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
                "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                "```python\n# YOUR CODE HERE\n```"
            )

        # Construct test cases
        public_test_cases = json.loads(example["public_test_cases"])
        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except Exception as e:
            print(f"Error loading private test cases: {e}")
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8"))))
            )
        full_test_cases = public_test_cases + private_test_cases

        metadata = json.loads(example["metadata"])
        test_cases = {
            "inputs": [t["input"] for t in full_test_cases],
            "outputs": [t["output"] for t in full_test_cases],
            "fn_name": metadata.get("func_name", None),
        }
        text_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
        return query_prompt, text_cases_compressed

    data_source = "livecodebench/code_generation_lite"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    try:
        dataset = load_dataset(data_source, split="test", trust_remote_code=True)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        raise RuntimeError(
            "LiveCodeBench uses a Hugging Face dataset script. Your installed `datasets` version rejected dataset "
            "scripts even with `trust_remote_code=True`. Create the LiveCodeBench parquet in an environment with "
            "`datasets<4.0.0`, for example: `pip install 'datasets<4.0.0'`, then rerun "
            "`python recipe/r1/data_process.py --local_dir ./data/r1_livecodebench --tasks livecodebench`."
        ) from exc
    # R1 Evaluation use LiveCodeBench 24.08-25.01
    dataset = dataset.filter(lambda line: "2024-08-00T00:00:00" <= line["contest_date"] < "2025-01-00T00:00:00")
    map_fn = partial(
        example_map_fn, process_fn=process_livecodebench, data_source=data_source, ability="Code", split="test"
    )

    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names, num_proc=8)
    return dataset


EVALPLUS_SYSTEM_PROMPT = (
    "You are a helpful programming assistant. The user will ask you a question and you as the assistant solve it. "
    "The assistant first thinks how to solve the task through reasoning and then provides the user with the final "
    "answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, "
    "respectively."
)


def _build_evalplus_dataset(dataset_name):
    from datasets import Dataset
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset_name == "humanevalplus":
        data_source = "evalplus/humanevalplus"
        problems = get_human_eval_plus()
    elif dataset_name == "mbppplus":
        data_source = "evalplus/mbppplus"
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unsupported EvalPlus dataset: {dataset_name}")

    print(f"Loading the {data_source} dataset from evalplus...", flush=True)
    rows = []
    for idx, (task_id, problem) in enumerate(problems.items()):
        query_prompt = (
            "Solve the following Python programming task. Return a self-contained Python solution in a markdown "
            f"code block.\n\n{problem['prompt']}"
        )
        ground_truth = {
            "task_id": task_id,
            "entry_point": problem["entry_point"],
            "prompt": problem["prompt"],
            "canonical_solution": problem["canonical_solution"],
            "base_input": problem["base_input"],
            "plus_input": problem["plus_input"],
            "atol": problem.get("atol", 1e-6),
        }
        rows.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": EVALPLUS_SYSTEM_PROMPT},
                    {"role": "user", "content": query_prompt},
                ],
                "ability": "Code",
                "reward_model": {"style": "rule", "ground_truth": _serialize_evalplus_ground_truth(ground_truth)},
                "extra_info": {"split": "test", "index": idx, "task_id": task_id, "dataset": data_source},
            }
        )

    return Dataset.from_list(rows)


def _serialize_evalplus_ground_truth(ground_truth):
    try:
        return json.dumps(ground_truth)
    except TypeError:
        payload = base64.b64encode(zlib.compress(pickle.dumps(ground_truth))).decode("utf-8")
        return json.dumps({"__evalplus_pickle__": payload})


def build_humanevalplus_dataset():
    return _build_evalplus_dataset("humanevalplus")


def build_mbppplus_dataset():
    return _build_evalplus_dataset("mbppplus")


TASK2DATA = {
    "aime2024": build_aime2024_dataset,
    "gpqa_diamond": build_gpqa_dimond_dataset,
    "cnmo2024": build_cnmo2024_dataset,
    "livecodebench": build_livecodebench_dataset,
    "humanevalplus": build_humanevalplus_dataset,
    "mbppplus": build_mbppplus_dataset,
}
SUPPORTED_TASKS = TASK2DATA.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/r1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tasks", default="all")

    args = parser.parse_args()

    if args.tasks.lower() == "all":
        args.tasks = SUPPORTED_TASKS
    else:
        args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        for task in args.tasks:
            if task not in SUPPORTED_TASKS:
                raise NotImplementedError(f"{task} has not been supported.")

    datasets = []
    for task in args.tasks:
        datasets.append(TASK2DATA[task]())
    test_dataset = concatenate_datasets(datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    to_parquet_compat(test_dataset, os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
