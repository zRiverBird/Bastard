### 1. 拼接六视图图像

运行脚本 `concat_6_views.sh`，根据你的数据路径调整：

```bash
QA_JSON_DIR=/你的/qa_json/文件夹路径
INPUT_PIC_PATH=/你的/原始图像文件夹路径
OUTPUT_DIR=/你的/输出拼接图路径
N_PROCESS=8  # 根据机器核数调整
```

### 2. LLaMA 格式 JSON 转换

运行脚本 `convert2llama.sh`，根据你的数据路径调整：
``` bash

INPUT_JSON=/你的/输入/json文件路径
OUTPUT_JSON=/你的/输出/json文件路径
IS_TRAIN=false  # true 表示训练集，false 表示验证集 （后面更新现在没用）

```