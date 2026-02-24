from pathlib import Path

from lighteval.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.aime import aime_prompt
from lighteval.tasks.tasks.gpqa import gpqa_instruct_prompt
from lighteval.tasks.tasks.gsm8k import gsm8k_prompt
from lighteval.tasks.tasks.ifeval.main import ifeval_metrics, ifeval_prompt
from lighteval.tasks.tasks.math_500 import math_500_prompt
from lighteval.tasks.tasks.mmlu_pro import mmlu_pro_prompt_function
from lighteval.tasks.tasks.lcb.main import lcb_codegeneration_prompt_fn


# Use an absolute path string here.
DATA_ROOT_TEXT = r"D:\workspace\agent\lighteval\offline_datasets_selected"
DATA_ROOT = Path(DATA_ROOT_TEXT)


local_gsm8k = LightevalTaskConfig(
    name="local_gsm8k",
    prompt_function=gsm8k_prompt,
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files={"test": str(DATA_ROOT / "gsm8k" / "*.arrow")},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=["Question:"],
    version=0,
)

local_gpqa_diamond = LightevalTaskConfig(
    name="local_gpqa_diamond",
    prompt_function=gpqa_instruct_prompt,
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files={"train": str(DATA_ROOT / "gpqa_diamond" / "*.arrow")},
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 1})],
    stop_sequence=[],
    version=0,
)

local_mmlu_pro = LightevalTaskConfig(
    name="local_mmlu_pro",
    prompt_function=mmlu_pro_prompt_function,
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files={"test": str(DATA_ROOT / "mmlu_pro" / "*.arrow")},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],
    version=0,
)

local_math_500 = LightevalTaskConfig(
    name="local_math_500",
    prompt_function=math_500_prompt,
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files={"test": str(DATA_ROOT / "math_500" / "*.arrow")},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1})],
    version=0,
)

local_aime24 = LightevalTaskConfig(
    name="local_aime24",
    prompt_function=aime_prompt,
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files={"train": str(DATA_ROOT / "aime24" / "*.arrow")},
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=0,
)

local_ifeval = LightevalTaskConfig(
    name="local_ifeval",
    prompt_function=ifeval_prompt,
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files={"train": str(DATA_ROOT / "ifeval" / "*.arrow")},
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    metrics=[ifeval_metrics],
    stop_sequence=[],
    version=0,
)

local_livecodebench = LightevalTaskConfig(
    name="local_livecodebench",
    prompt_function=lcb_codegeneration_prompt_fn,
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files={"test": str(DATA_ROOT / "livecodebench" / "*.arrow")},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.lcb_codegen_metric],
    stop_sequence=[],
    version=0,
)

TASKS_TABLE = [
    local_gsm8k,
    local_gpqa_diamond,
    local_mmlu_pro,
    local_math_500,
    local_aime24,
    local_ifeval,
    local_livecodebench,
]
