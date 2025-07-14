import json
import random
from tqdm import tqdm
from pprint import pprint as pp

# single image
# cams = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
# prompts = open("prompts.txt").readlines()
# lines = open("caption.jsonl").readlines()
# res = open("16g_caption.jsonl", 'w')
# n = 0
# for line in tqdm(lines):
#     data = json.loads(line)
#     k = list(data.keys())[0]
#     v = list(data.values())[0]
#     image_paths = v["image_paths"]
#     answers = v["answer"]

#     for cam in cams:
#         path = image_paths[cam].split('/', maxsplit=3)[-1]
#         caption = answers[cam]

#         temp = {}
#         temp["id"] = f"{k}_{cam}"
#         temp["image"] = "/aojidata-sh/llm/Qwen2.5-VL/qwen-vl-finetune/data/16g/nuscenes/samples/" + path
#         temp["conversations"] = [
#                 {"from": "human", "value": "<image>\n" + random.choice(prompts).strip()},
#                 {"from": "gpt", "value": caption},
#             ]

#         res.write(json.dumps(temp) + '\n')



# 6 concat image
cams = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
prompts = open("prompts.txt").readlines()
lines = open("caption.jsonl").readlines()
res = open("16g_caption_concat.jsonl", 'w')
n = 0
for line in tqdm(lines):
    data = json.loads(line)
    k = list(data.keys())[0]
    v = list(data.values())[0]
    concat_image_path = v["concat_image_path"]
    answers = v["answer"]

    temp = {}
    temp["id"] = f"{k}_ALL_CONCAT"
    temp["image"] = concat_image_path
    temp["conversations"] = [{"from": "human", "value": "<image>\n" + random.choice(prompts).strip()}]
    caption = "This is an image composed of six camera views around a car. The layout is as follows: the first row contains three images: the left front, front, and right front images of the car. The second row contains three images: the left rear, rear, and right rear images of the car. I will analyze the contents of each of the six images in order.\n"
    for cam in cams:
        cur_caption = answers[cam]
        cur_caption = f"{cam}: {cur_caption}\n"
        caption += cur_caption

    temp["conversations"].append({"from": "gpt", "value": caption})

    res.write(json.dumps(temp) + '\n')












