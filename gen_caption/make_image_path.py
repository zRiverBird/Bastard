import json
import os
from tqdm import tqdm

res = {}
n = 0
data = json.load(open("/aojidata-sh/llm/Qwen2.5-VL/qwen-vl-finetune/data/16g/QA_data_set_nus/output.json"))
for k, v in tqdm(data.items()):
    frames = v["key_frames"]
    for frame, info in frames.items():
        res_k = f"{k}_{frame}"
        image_paths = info["image_paths"]
        concat_image_path = info["concat_image_path"]

        res[res_k] = {}
        res[res_k]["image_paths"] = image_paths
        res[res_k]["concat_image_path"] = concat_image_path

        n += 1

print(n)
json.dump(res, open("images.json",'w'), indent=2, ensure_ascii=False)




