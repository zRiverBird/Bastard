set -x

#model="/aojidata-sh/llm/Qwen2.5-VL/qwen-vl-finetune/output/16g/4sp_sys"
model="/aojidata-sh/llm/internvl3/internvl_chat/work_dirs/16g/5sys"

vllm serve $model --allowed-local-media-path / --tensor-parallel-size 1
