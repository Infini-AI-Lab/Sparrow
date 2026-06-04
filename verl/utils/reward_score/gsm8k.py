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

import re

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1] 

_SOLUTION_CLIP_CHARS = 300

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval 

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer 

def extract_solution2(solution_str):
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str) 
        print("string_in_last_boxed {} string_in_last_boxed type {}".format(string_in_last_boxed, type(string_in_last_boxed))) 
        if string_in_last_boxed is not None:
            return remove_boxed(string_in_last_boxed)
        else:
            return None
    except Exception as e:
        print(e)
        print("@@@"*20)
        print(solution_str)
        print("+++"*20)
        print("Failed to extract answer from solution string.")
        print("@@@"*20)
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method) 
    answeranother = extract_solution2(solution_str=solution_str) 
    print("answer {} answer type {} answeranother {} answeranother type {} ground_truth {} ground_truth type {}".format(answer, type(answer), answeranother, type(answeranother), ground_truth, type(ground_truth))) 
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score 

if __name__ == "__main__": 
    solution_str = "To solve this problem, let's break it down step by step.                                                                      \
(TaskRunner pid=46215)                                                                                                                               \
(TaskRunner pid=46215) Step 1: Determine the total number of slices.                                                                                 \
(TaskRunner pid=46215) There are 7 pizzas, and each pizza is cut into 8 slices. So, the total number of slices is:                                   \
(TaskRunner pid=46215) 7 pizzas * 8 slices/pizza = 56 slices                                                                                         \
(TaskRunner pid=46215)                                                                                                                               \
(TaskRunner pid=46215) Step 2: Determine the total number of people.                                                                                 \
(TaskRunner pid=46215) Henry and 3 of his friends are ordering the pizzas, so there are a total of:                                                  \
(TaskRunner pid=46215) 1 (Henry) + 3 (friends) = 4 people                                                                                            \
(TaskRunner pid=46215)                                                                                                                               \
(TaskRunner pid=46215) Step 3: Determine the number of slices each person can have.                                                                  \
(TaskRunner pid=46215) To share the pizzas equally, we need to divide the total number of slices by the total number of people:                      \
(TaskRunner pid=46215) 56 slices / 4 people = 14 slices per person                                                                                   \
(TaskRunner pid=46215)                                                                                                                               \
(TaskRunner pid=46215) Therefore, each of Henry and his friends can have 14 slices.                                                                  \
(TaskRunner pid=46215)                                                                                                                               \
(TaskRunner pid=46215) \\boxed{14}" 

    print(extract_solution2(solution_str)) 