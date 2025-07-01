#!/bin/bash

# 确保脚本出错立即停止
set -e

# 修改为你自己的路径
QA_JSON_DIR=/home/b311/data/zjp/DriveLM/data/QA_data_set_nus/
INPUT_PIC_PATH=/home/b311/data/zjp/DriveLM/data/nuscenes/samples/
OUTPUT_DIR=/home/b311/data/zjp/DriveLM/data/nuscenes/samples/ALL_CONCAT/
N_PROCESS=8

python tools/concat_6_views.py \
  --qa_json_dir ${QA_JSON_DIR} \
  --input_pic_path ${INPUT_PIC_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --n_process ${N_PROCESS}
