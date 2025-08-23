# VeRL

## 数据准备 Data Preparation

[Prepare Data for Post-Training — verl documentation](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)

核心思路：将数据集做成有如下几个字段的列表：

```
1. data_source
作用: 数据集的唯一标识符，用于后续匹配对应的奖励函数。
格式: 字符串 (String)。
示例: 'openai

2. prompt
作用: 将要输入给大模型的实际内容。
格式: Hugging Face chat_template 格式（一个字典列表）。
角色 (role):
"user": 代表用户的直接输入。
"system": (可选) 用于提供高层次的系统级指令或设定模型的“人设”，通常放在列表开头。
"assistant": (可选) 用于多轮对话中模型的回复。

示例:
[
    { "role": "system", "content": "你是一位专业的代码助手。" },
    { "role": "user", "content": "用 Python 写一个快速排序算法。" }
]

3. ability
作用: 定义任务所属的类别。
格式: 字符串 (String)。
示例: 'math', 'coding', 'translation', 'summarization'。

4. reward_model
作用: 存放用于计算奖励 (Reward) 的信息。
格式: 字典 (Dictionary)。
内部字段:
style: 奖励计算的方式。示例值为 "rule"，表示基于规则匹配。
ground_truth: 标准答案。其数据类型灵活，取决于你的任务。你提供的 ground_truth 必须能被你的奖励函数正确解析。

示例:
{
    "style": "rule",
    "ground_truth": "15"
}

5. extra_info
作用: 记录与训练无关的额外信息，主要用于调试和数据溯源。
格式: 字典 (Dictionary)。
示例: {'split': 'train', 'index': 42}。
```

然后转化为**.parquet**格式即可



## 自定义奖励函数

### 运行命令中自定义

- 函数签名要求：

```python
def your_function_name(data_source, solution_str, ground_truth, extra_info=None):
    pass
```

必须包含data_source, solution_str, ground_truth, extra_info四个字段，与数据部分的含义相同



- 命令行参数设置：

```
python -m verl.trainer.main_ppo \
  ... \
  custom_reward_function.path=my_rewards.py \
  custom_reward_function.name=my_amazing_reward_fn \
  ...
```

需要指定py文件的路径和函数名

### 框架内置

`/verl/utils/reward_score`里实现了很多demo的奖励函数，如`gsm8k.py`就实现了QuickStart中的demo的奖励函数：

```python
import re

_SOLUTION_CLIP_CHARS = 300


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
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
```

再通过`/verl/utils/reward_score/__init__.py`来注册：注意是根据data_source字段值的不同来判断的

```python
def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])
```

如果要自定义奖励函数，也可以考虑框架在`/verl/utils/reward_score`下开一个py文件，然后在`__init__.py`中的if-else逻辑中指定对应数据集的data_source，判断对应奖励函数

**（来自gemini）总结一下**：除非你有特殊理由，否则**强烈推荐你使用 `custom_reward_function.path` 和 `custom_reward_function.name` 这两个运行时参数**。这是最干净、最方便的做法。

### 补充信息

1. 一个便捷方式

如果你在自定义的 `.py` 文件中，将你的奖励函数**直接命名为 `compute_score`**，那么你在运行命令时就**不需要设置 `custom_reward_function.name` 这个参数了**。

**示例：** 如果你的 `my_rewards.py` 文件内容是：

```python
def compute_score(solution_str, ground_truth, ...):
    # 你的逻辑
    return score
```

那么你的启动命令就可以简化为：

```bash
python3 -m verl.trainer.main_ppo \
  ... \
  custom_reward_function.path=my_rewards.py \
  ...
```

2. 便于实验、A/B测试的最佳实践

你可以在**同一个 `.py` 文件里定义多个不同的奖励函数**。

这样一来，当你想切换不同的打分逻辑进行实验时，你只需要修改启动命令中的 `custom_reward_function.name` 这一个参数，而不需要更改文件路径，非常方便。

**示例：** 你的 `my_rewards.py` 文件可以这样写：

```python
def reward_logic_v1(solution_str, ...):
    # 版本1的打分逻辑
    return score

def reward_logic_v2(solution_str, ...):
    # 版本2的打分逻辑
    return score
```

然后你可以轻松地在两次不同的实验中切换：

- **实验 A**: `... custom_reward_function.name=reward_logic_v1 ...`
- **实验 B**: `... custom_reward_function.name=reward_logic_v2 ...`

3. 奖励函数的不同类型

最后，手册也明确了奖励函数的来源可以是多种多样的，不仅仅是基于规则的代码：

- **规则型**: 就像 GSM8k 的例子，通过字符串匹配和正则表达式来打分。
- **模型型**: 对于 RLHF 数据集，明确提到会使用一个**奖励模型 (Reward Model)** 来打分。
- **沙箱型**: 对于代码生成任务，会使用 **SandBox (沙箱)** 来实际执行代码并根据测试用例的通过情况来打分。

这为你实现自己的 `compute_score` 函数提供了更广阔的思路：你的函数内部不仅可以是简单的 `if/else`，还可以是调用一个外部 API、一个预训练模型，或者一个代码执行环境。

