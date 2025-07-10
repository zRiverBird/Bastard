set -x

MODEL_PATH=""/aojidata-sh/llm/internvl3/internvl_chat/pretrain/InternVL3-8B/""
INPUT_DATA="./images.json"
OUTPUT_DIR="./"
MAX_MODEL_LEN=8192
TEMPERATURE=0
TOP_P=0.9
MAX_TOKENS=512
PORT=8000

mkdir -p ${OUTPUT_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/caption.json"

echo "Running inference..."
python inference.py \
    --model ${MODEL_PATH} \
    --data ${INPUT_DATA} \
    --output ${OUTPUT_FILE} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --api_base "http://11.131.245.158:${PORT}/v1"
