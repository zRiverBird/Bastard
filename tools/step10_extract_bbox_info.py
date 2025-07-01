import os
import pickle
import tqdm
import numpy as np

from multiprocessing import Process, Manager, cpu_count

def local2global(camera_name, coord):
    new_x = coord[0] * resize_num[0]
    new_y = coord[1] * resize_num[1]
    if camera_name == "CAM_FRONT_LEFT":
       pass
    if camera_name == "CAM_FRONT":
        new_x =  new_x + new_image_size[0]
    if camera_name == "CAM_FRONT_RIGHT":
        new_x = new_x + new_image_size[0]*2
    if camera_name == "CAM_BACK_LEFT":
        new_y = new_y + new_image_size[1]
    if camera_name == "CAM_BACK":
        new_y = new_y + new_image_size[1]
        new_x = new_x + new_image_size[0]
    if camera_name == "CAM_BACK_RIGHT":
        new_y = new_y + new_image_size[1]
        new_x = new_x + new_image_size[0]*2
    new_x = int(new_x / global_image_size[0])
    new_y = int(new_y / global_image_size[1])
    return np.array([new_x,new_y])

def get_box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def get_box_dict(bboxes2d, cam):
    center_dict = []
    for box in bboxes2d:
        center = get_box_center(box)
        new_center = local2global(cam, center).tolist()
        new_tl_br = local2global(cam, box[:2]).tolist() + local2global(cam, box[2:]).tolist()
        bbox_dict = {"center_info" : new_center, "bbox_info" : new_tl_br}
        center_dict.append(bbox_dict)
    return center_dict

def get_info_cam_dict():
    total_dict = {}
    for current_id in tqdm.tqdm(range(len(key_infos['infos']))):
        token = key_infos['infos'][current_id]['token']
        bboxes2d = key_infos['infos'][current_id]['bboxes2d']
        centers2d = key_infos['infos'][current_id]['centers2d']
        labels2d = key_infos['infos'][current_id]['labels2d']
        gt_names = key_infos['infos'][current_id]['gt_names']

        total_dict[token] = {}
        total_dict[token]['labels2dnum'] = {}
        total_dict[token]['labels2dname'] = {}
        for i, label in enumerate(labels2d):
            name_list = []
            total_dict[token]['labels2dnum'][cams[i]] = label.copy() 
            for label_num in list(label):
                name_list.append(label_name[label_num])
            total_dict[token]['labels2dname'][cams[i]] = name_list.copy()

        total_dict[token]['local_bboxes2d'] = {}
        total_dict[token]['global_center_bbox'] = {}
        for i, bboxes in enumerate(bboxes2d):
            center_dict = get_box_dict(bboxes, cams[i])
            total_dict[token]['local_bboxes2d'][cams[i]] = bboxes.copy()
            total_dict[token]['global_center_bbox'][cams[i]] = center_dict.copy()
        
        total_dict[token]['local_center'] = {}
        total_dict[token]['global_center_v1'] = {}
        for i, center in enumerate(centers2d):
            center_org = []
            if len(center)>0:
                for center_ in center:
                    center_org.append(local2global(cams[i], center_).tolist())
            total_dict[token]['local_center'][cams[i]] = center.copy()
            total_dict[token]['global_center_v1'][cams[i]] = center_org.copy()
            
        total_dict[token]['camera_infos'] = key_infos['infos'][current_id]['cams']
    return total_dict

if __name__ == '__main__':
    pkl_data_root = "./data/nuscenes/"

    info_prefix = 'train'
    key_infos = pickle.load(open(os.path.join(pkl_data_root,'nuscenes2d_ego_temporal_infos_{}.pkl'.format(info_prefix)), 'rb'))
    cams = list(key_infos['infos'][0]['cams'].keys())
    
    colors = {"CAM_FRONT_LEFT": (0, 0, 255), "CAM_FRONT": (0, 255, 0), "CAM_FRONT_RIGHT": (255, 0, 0), "CAM_BACK_LEFT": (0, 255, 255),
            "CAM_BACK": (255, 255, 0), "CAM_BACK_RIGHT": (255, 0, 255)}
    label_name= ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

    new_image_size = [896, 448]
    ori_image_size = [1600, 900]
    global_image_size = [1, 1] # 不加入1000坐标转换
    # global_image_size = [new_image_size[0] * 3 / 1000, new_image_size[1] * 2 / 1000] # 加入1000坐标转换
    resize_num = (round(new_image_size[0] / ori_image_size[0],3), round(new_image_size[1] / ori_image_size[1],3)) # w, h

    num_processes = 20 # 你可以根据需要设置进程数
    manager = Manager()
    save_dict = manager.list()
    mutil_process_total_list = []
    total_dict = get_info_cam_dict()
    for index, (key, value) in enumerate(total_dict.items()):
        mutil_process_total_list.append([key, value])
        # if index == 50:
        #     break
    print(len(mutil_process_total_list))
    # processes = []
    # for i in range(num_processes):
    #     start_index = i * len(mutil_process_total_list) // num_processes
    #     end_index = (i + 1) * len(mutil_process_total_list) // num_processes
    #     p = Process(target=process_data, args=(mutil_process_total_list, i, start_index, end_index, save_dict, dst_concat_image_root))
    #     processes.append(p)
    #     p.start()

    #   # 等待所有进程结束
    # for p in processes:
    #     p.join()