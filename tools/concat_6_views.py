import json
import cv2
import numpy as np
import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm

ordered_keys = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
]

colors = {
    "CAM_FRONT_LEFT": (0, 0, 255),
    "CAM_FRONT": (0, 255, 0),
    "CAM_FRONT_RIGHT": (255, 0, 0),
    "CAM_BACK_LEFT": (0, 255, 255),
    "CAM_BACK": (255, 255, 0),
    "CAM_BACK_RIGHT": (255, 0, 255)
}

def make_cat_image(images):
    concat_images = np.zeros([896, 2688, 3], dtype=np.uint8)
    positions = {
        "CAM_FRONT_LEFT": (0, 0),
        "CAM_FRONT": (0, 896),
        "CAM_FRONT_RIGHT": (0, 1792),
        "CAM_BACK_LEFT": (448, 0),
        "CAM_BACK": (448, 896),
        "CAM_BACK_RIGHT": (448, 1792)
    }

    for key in ordered_keys:
        img = cv2.imread(images[key])
        img = cv2.putText(img, key, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[key], 2, cv2.LINE_AA)
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=colors[key])
        img = cv2.resize(img, (896, 448))
        y, x = positions[key]
        concat_images[y:y+448, x:x+896] = img

    return concat_images

def process_single(args):
    scene_id, token, image_paths, input_path, output_path = args
    img_list = {}
    for key in ordered_keys:
        img_path = image_paths[key].replace("../nuscenes/samples/", input_path)
        img_list[key] = img_path

    final_image = make_cat_image(img_list)
    output_pic_path = os.path.join(output_path, f"{scene_id}_{token}.png")
    cv2.imwrite(output_pic_path, final_image)

    return (scene_id, token, output_pic_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多进程拼接六视图图像，带进度条")
    parser.add_argument('--qa_json_dir', type=str, required=True, help='QA json 文件夹路径')
    parser.add_argument('--input_pic_path', type=str, required=True, help='原始图像文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='拼接结果输出路径')
    parser.add_argument('--n_process', type=int, default=8, help='并行进程数')

    args = parser.parse_args()

    json_path = os.path.join(args.qa_json_dir, "v1_1_train_nus.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    task_list = []
    for scene_id in data.keys():
        for token in data[scene_id]['key_frames']:
            image_paths = data[scene_id]['key_frames'][token]['image_paths']
            task_list.append((scene_id, token, image_paths, args.input_pic_path, args.output_dir))

    results = []
    with Pool(processes=args.n_process) as pool:
        for result in tqdm(pool.imap_unordered(process_single, task_list), total=len(task_list), desc="Processing"):
            results.append(result)

    for scene_id, token, output_pic_path in results:
        data[scene_id]['key_frames'][token]['concat_image_path'] = output_pic_path

    output_json_name = "output.json"
    with open(os.path.join(args.qa_json_dir, output_json_name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)