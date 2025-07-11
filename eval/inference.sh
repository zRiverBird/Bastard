set -x

#MODEL_PATH="/aojidata-sh/llm/Qwen2.5-VL/qwen-vl-finetune/output/16g/4sp_sys"
MODEL_PATH="/aojidata-sh/llm/internvl3/internvl_chat/work_dirs/16g/5sys"
INPUT_DATA="jsons/phase1_test/robosense_track1_release_convert_concat6.json"
OUTPUT_DIR="/aojidata-sh/llm/Bastard/test_res/"
MAX_MODEL_LEN=8192
TEMPERATURE=0
TOP_P=0.9
MAX_TOKENS=512
PORT=8000

mkdir -p ${OUTPUT_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.json"

echo "Running inference..."
python inference.py \
    --model ${MODEL_PATH} \
    --data ${INPUT_DATA} \
    --output ${OUTPUT_FILE} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --api_base "http://127.0.0.1:${PORT}/v1"
