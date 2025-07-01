#!/bin/bash

INPUT_JSON="data/QA_data_set_nus/output.json"
OUTPUT_JSON="data/QA_data_set_nus/output-llama.json"
IS_TRAIN=false  # 改为 true 表示训练集，影响路径替换

if [ "$IS_TRAIN" = true ]; then
    python tools/convert2llama.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" --is_train
else
    python tools/convert2llama.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON"
fi
