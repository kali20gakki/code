import json, os
import numpy as np
import cv2
import warnings
import random

def paste_obj2img(img, obj, seg, coor, new_h):
    """ 把一个目标贴到图片指定位置
    Args:
        img ([np.array]): [H, W, 3]
        obj ([np.array]): [H, W, 3]
        seg ([np.array]): [[x1, y1], [x2, y2],...]
        coor ([type]): (x1, y1)
        new_h ([int]): h
    """
    img_h, img_w, img_c = img.shape
    obj_h, obj_w, obj_c = obj.shape

    if (img_c != obj_c):
        warnings.warn('img channel != obj channel')
        return img, None

    if(img_h < obj_h or img_w < obj_w):
        warnings.warn("img size < obj size")
        return img, None

    if(((coor[0]+obj_w)>img_w) or ((coor[1]+obj_h)>img_h)):
        warnings.warn("coor + obj out of Img range")
        return img, None

    is_flip = False
    if random.randint(1,10) > 6:
        is_flip = True

    out_img = img.copy()
    # 1. resize keep ratio
    ratio = float(obj_h / obj_w)
    new_w = int(new_h / ratio)
    ratio_x = float(new_w / obj_w)
    ratio_y = float(new_h / obj_h)

    resized_obj = cv2.resize(obj, (new_w, new_h))
    obj_array=np.zeros((new_h,new_w,obj_c),dtype=np.uint8) # obj大小
    seg = np.array([[int(coor[0] * ratio_x), int(coor[1] * ratio_y)] for coor in seg]) # resize seg
    coor = list(map(int, coor))
    input_roi = out_img[coor[1]:coor[1]+new_h,coor[0]:coor[0]+new_w] # img上coor位置obj大小

    # 概率翻转
    if is_flip:
        cv2.fillPoly(obj_array, [seg], color=(1,1,1))
        seg = np.array([[int(new_w - coor[0]), coor[1]] for coor in seg]) # flip seg
        cv2.fillPoly(input_roi, [seg], color=(0, 0, 0))
        obj_array=cv2.flip(obj_array,1)
        resized_obj=cv2.flip(resized_obj,1)
    else:
        cv2.fillPoly(obj_array, [seg], color=(1,1,1))
        cv2.fillPoly(input_roi, [seg], color=(0, 0, 0))

    obj_array = obj_array * resized_obj # obj抠图
    obj_array = input_roi + obj_array # obj roi + img环境
    
    # 贴图
    out_img[coor[1]:coor[1] + new_h, coor[0]:coor[0] + new_w] = obj_array

    # 计算bbox x1, y1, w, h
    bbox = [coor[0], coor[1], coor[0] + new_w, coor[1] + new_h]
    return out_img, bbox


def get_polygon(seg_file_path):
    with open(seg_file_path, 'r') as f:
        seg_ann = json.load(f)
    polygon = seg_ann['outputs']['object'][0]['polygon']
    polygon = np.array([polygon[coor] for coor in polygon]).reshape([-1, 2])
    return polygon


def select_points(gt_bbox, img_shape, num_points):
    """[summary]
    Args:
        gt_bbox ([type]): [[x1, y1, x2, y2], ...]
        img_shape ([type]): [H, W]
        bbox_size: [[W, H], ...]
        num_points: 2, 4, 8
    """
    MAX_W, MAX_H = 96, 96
    # 范围 0.25*W < x < 0.75*W
    # 筛选出在贴图范围内的gt
    H, W = img_shape
    boundary_left = int(0.25 * W)
    boundary_right = int(0.75 * W)
    in_left_gt = []
    in_right_gt = []
    for bbox in gt_bbox:
        if bbox[0] < boundary_left:
            in_left_gt.append(bbox)
        elif bbox[2] > boundary_right:
            in_right_gt.append(bbox)

    times = num_points//2
    # 贴左边
    left_points = []
    if len(in_left_gt) !=0: # 有gt
        bottom = [0, 0, 0, 0] # 最底部的gt
        for bbox in in_left_gt:
            if bbox[3] > bottom[3]:
                bottom = bbox

        if bottom[3] <= int(0.35*H):   # 图的上半部分
            for i in range(times): # 大致水平排列
                ref_x = MAX_W * i + random.randint(1, 15)
                ref_y = bottom[3] + random.randint(20, 50)
                left_points.append([ref_x, ref_y])
        elif bottom[3] >= int(0.65*H): # 图的下半部分
            for i in range(times): # 大致水平排列
                ref_x = MAX_W * i + random.randint(1, 15)
                ref_y = int(0.35*H) + random.randint(20, 50)
                left_points.append([ref_x, ref_y])
        else:                          # 图的中间部分
            for i in range(times): # 大致水平排列
                ref_x = MAX_W * i + random.randint(1, 15)
                ref_y = int(0.75*H) + random.randint(10, 20)
                left_points.append([ref_x, ref_y])

    else:# 无gt 均匀贴图
        y_interval = int(H / times)
        for i in range(times):
            ref_x = random.randint(1, boundary_left-MAX_W)
            ref_y = random.randint(y_interval*i, y_interval*(i+1)-MAX_H)
            left_points.append([ref_x, ref_y])

    # 贴右边
    right_points = []
    if len(in_right_gt) !=0: # 有gt
        bottom = [0, 0, 0, 0] # 最底部的gt
        for bbox in in_right_gt:
            if bbox[3] > bottom[3]:
                bottom = bbox

        if bottom[3] <= int(0.35*H):   # 图的上半部分
            for i in range(times): # 大致水平排列
                ref_x = MAX_W * i + boundary_right + random.randint(10, 20)
                ref_y = bottom[3] + random.randint(20, 50)
                right_points.append([ref_x, ref_y])
        elif bottom[3] >= int(0.65*H): # 图的下半部分
            for i in range(times): # 大致水平排列
                ref_x = MAX_W * i + boundary_right + random.randint(10, 20)
                ref_y = int(0.35*H) + random.randint(20, 50)
                right_points.append([ref_x, ref_y])
        else:                          # 图的中间部分
            for i in range(times): # 大致水平排列
                ref_x = MAX_W * i + boundary_right + random.randint(10, 20)
                ref_y = int(0.75*H) + random.randint(10, 20)
                right_points.append([ref_x, ref_y])
    else:# 无gt 均匀贴图
        y_interval = int(H / times)
        for i in range(times):
            ref_x = random.randint(boundary_right, W-MAX_W)
            ref_y = random.randint(y_interval*i, y_interval*(i+1)-MAX_H)
            right_points.append([ref_x, ref_y])
    
    return left_points+right_points