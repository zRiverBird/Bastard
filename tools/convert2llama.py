import os
import json
import argparse
from tqdm import tqdm

def convert2llama(input_json, output_json, is_train):
    """
    将 QA 数据转换为 llama 格式。

    Args:
        input_json (str): 输入 JSON 文件路径
        output_json (str): 输出 llama 格式 JSON 文件路径
        is_train (bool): 是否为训练集（影响路径替换）
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    results = []
    for scene_id, scene_data in tqdm(data.items(), desc="Processing Scenes"):
        for frame_id, frame_data in scene_data["key_frames"].items():
            image_paths_dict = frame_data["concat_image_path"]

            # # 路径处理
            # if is_train:
            #     image_paths = [
            #         os.path.join(
            #             "/vepfs_for_algorithm_from_nas/nas_fsd/vePFS_test/vepfs_fsd/nas_fsd3/wsy/code/other_code/DriveLM-main/challenge",
            #             path.replace("..", "data")
            #         ) for path in image_paths_dict.values()
            #     ]
            # else:
            #     image_paths = [
            #         os.path.join(
            #             "/vepfs_for_algorithm_from_nas/nas_fsd/vePFS_test/vepfs_fsd/nas_fsd3/wsy/code/other_code/DriveLM-main/challenge",
            #             path.replace("../nuscenes/samples", "data/nuscenes/val_data")
            #         ) for path in image_paths_dict.values()
            #     ]

            # 整理 QA
            qa_data = frame_data["QA"]
            qa_list = qa_data["perception"] + qa_data["prediction"] + qa_data["planning"] + qa_data["behavior"]

            for idx, qa in enumerate(qa_list):
                results.append({
                    "id": f"{scene_id}_{frame_id}_{idx}",
                    "image": image_paths_dict,
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + qa["Q"]},
                        {"from": "gpt", "value": qa["A"]}
                    ]
                })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[完成] 共生成 {len(results)} 条数据，保存至 {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriveLM 数据转换为 llama 格式")
    parser.add_argument('--input_json', type=str, required=True, help='输入 JSON 文件路径')
    parser.add_argument('--output_json', type=str, required=True, help='输出 JSON 文件路径')
    parser.add_argument('--is_train', action='store_true', help='是否为训练集，决定路径替换规则')

    args = parser.parse_args()

    convert2llama(args.input_json, args.output_json, args.is_train)
