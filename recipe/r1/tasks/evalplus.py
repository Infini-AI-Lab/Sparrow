import base64
import json
import math
import pickle
import textwrap
import zlib

from verl.utils.reward_score import coder1

_MAX_LOG_CHARS = 2048


def _functional_check_code(ground_truth: dict) -> str:
    entry_point = ground_truth["entry_point"]
    prompt = ground_truth["prompt"]
    canonical_solution = ground_truth["canonical_solution"]
    reference_solution = canonical_solution
    if prompt.lstrip().startswith(("def ", "async def ", "class ")):
        reference_solution = prompt + canonical_solution
    test_inputs = _as_test_case_list(ground_truth["base_input"]) + _as_test_case_list(ground_truth["plus_input"])
    atol = ground_truth.get("atol", 1e-6)

    return f"""
import math

_candidate = {entry_point}

{reference_solution}

_reference = {entry_point}
_inputs = {test_inputs!r}
_atol = {atol!r}

def _deep_eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=_atol, abs_tol=_atol)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
    return a == b

for _case in _inputs:
    if isinstance(_case, list):
        _case = tuple(_case)
    elif not isinstance(_case, tuple):
        _case = (_case,)
    _expected = _reference(*_case)
    _actual = _candidate(*_case)
    assert _deep_eq(_actual, _expected), f"expected {{_expected!r}}, got {{_actual!r}}, input={{_case!r}}"
"""


def _as_test_case_list(test_inputs):
    if test_inputs is None:
        return []
    if isinstance(test_inputs, list):
        return test_inputs
    if isinstance(test_inputs, tuple):
        return list(test_inputs)
    if isinstance(test_inputs, dict):
        cases = []
        for value in test_inputs.values():
            if value is None:
                continue
            if isinstance(value, list) and all(isinstance(case, (list, tuple)) for case in value):
                cases.extend(value)
            else:
                cases.append(value)
        return cases
    return [test_inputs]


def compute_score(solution_str, ground_truth, extra_info=None, check_mode=None):
    ground_truth = _load_ground_truth(ground_truth)

    reward_log = []
    pass_fmt = coder1.validate_response_structure(solution_str)
    if not pass_fmt:
        reward_log.append("Bad format detected; attempting code-block extraction anyway")

    solution_code = coder1.extract_code_from_string(solution_str)
    if not solution_code:
        reward_log.append("No Python code block extracted")
        _maybe_log(0.0, reward_log)
        return 0.0

    check_code = solution_code + "\n" + textwrap.dedent(_functional_check_code(ground_truth))
    succ, output = coder1.code_exec(check_code, check_mode=check_mode)
    if succ:
        reward_log.append("EvalPlus base+plus tests passed")
        _maybe_log(1.0, reward_log)
        return 1.0

    reward_log.append("EvalPlus base+plus tests failed")
    reward_log.append(output[:_MAX_LOG_CHARS])
    if extra_info and "task_id" in extra_info:
        reward_log.append(f"task_id={extra_info['task_id']}")
    _maybe_log(0.0, reward_log)
    return 0.0


def _load_ground_truth(ground_truth):
    if not isinstance(ground_truth, str):
        return ground_truth
    decoded = json.loads(ground_truth)
    if "__evalplus_pickle__" not in decoded:
        return decoded
    return pickle.loads(zlib.decompress(base64.b64decode(decoded["__evalplus_pickle__"].encode("utf-8"))))


def _maybe_log(score, reward_log):
    if coder1._should_print_reward_log():
        print(f"EvalPlus reward = {score}\n" + "\n".join(reward_log) + "\n")
