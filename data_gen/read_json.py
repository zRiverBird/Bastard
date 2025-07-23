import os
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm

# 六视图顺序
ordered_keys = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
]

# 每个视图标注颜色
colors = {
    "CAM_FRONT_LEFT": (0, 0, 255),
    "CAM_FRONT": (0, 255, 0),
    "CAM_FRONT_RIGHT": (255, 0, 0),
    "CAM_BACK_LEFT": (0, 255, 255),
    "CAM_BACK": (255, 255, 0),
    "CAM_BACK_RIGHT": (255, 0, 255)
}

# 拼接六视图为一张图
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
        img_path = images.get(key)
        if not img_path or not os.path.exists(img_path):
            print(f"⚠️ Warning: missing image {key}: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to read {img_path}")
            continue

        img = cv2.putText(img, key, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[key], 2, cv2.LINE_AA)
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=colors[key])
        img = cv2.resize(img, (896, 448))

        y, x = positions[key]
        concat_images[y:y+448, x:x+896] = img

    return concat_images

# 主处理逻辑
def process_json(json_file, input_pic_path, output_path, output_json_file):
    print(f"📂 Loading JSON: {json_file}")
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    os.makedirs(output_path, exist_ok=True)
    
    # print(json_data)
    with open(output_json_file, 'w') as f:
        for item in tqdm(json_data, desc="📷 Processing"):
        
            token = list(item.keys())[0]
            images = item[token]
            
          
            img_dict = {key: images.get(key, "").replace("../nuscenes/samples", input_pic_path) for key in ordered_keys}
            content = {"image_paths": img_dict}

            concat_img = make_cat_image(img_dict)
            concat_img_path = os.path.join(output_path, f"{token}.png")
            cv2.imwrite(concat_img_path, concat_img)
            content["concat_image_path"] = concat_img_path

            json_line = json.dumps({token: content}, ensure_ascii=False)
            f.write(json_line + "\n")

        print(f"✅ Done! Saved to {output_json_file}")

    # 命令行入口
def main():
    parser = argparse.ArgumentParser(description="拼接六视图图像并保存JSON")
    parser.add_argument('--json_file', type=str, default="/home/b311/data/zjp/DriveLM/data_gen/nuscenes_infos_train.json", help='输入JSON路径')
    parser.add_argument('--input_pic_path', type=str, default="/home/b311/data/zjp/DriveLM/data/nuscenes/nuscenes/v1.0-trainval/samples", help='原始图像样本文件夹路径，例如：samples')
    parser.add_argument('--output_dir', type=str, default="./output_images", help='输出图像保存目录')
    parser.add_argument('--output_json', type=str, default="./concat_output.json", help='保存更新后的JSON路径')
    args = parser.parse_args()

    process_json(args.json_file, args.input_pic_path, args.output_dir, args.output_json)

if __name__ == '__main__':
    main()
