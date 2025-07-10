set -x

model="/aojidata-sh/llm/internvl3/internvl_chat/pretrain/InternVL3-8B/"

vllm serve $model --allowed-local-media-path / --tensor-parallel-size 1
