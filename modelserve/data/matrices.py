import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 定义YOLO格式标签的解析函数
def parse_yolo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            bboxes.append([class_id, x, y, w, h])
        return bboxes

# 定义从原图中获取目标区域图像的函数
def crop_objects(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = parse_yolo_label(label_path)
    objects = []
    for bbox in bboxes:
        class_id, x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        xmin = int((x - w / 2) * img_w)
        ymin = int((y - h / 2) * img_h)
        xmax = int((x + w / 2) * img_w)
        ymax = int((y + h / 2) * img_h)
        object_img = image[ymin:ymax, xmin:xmax]
        objects.append((object_img, class_id))
    return objects

# 定义resize函数，确保两个图像具有相同的尺寸
def resize_objects(objects, target_size):
    resized_objects = []
    for obj, class_id in objects:
        resized_obj = cv2.resize(obj, target_size)
        resized_objects.append((resized_obj, class_id))
    return resized_objects

# 定义SSIM计算函数
# 定义SSIM计算函数
def calculate_ssim(obj1, obj2, win_size=(5, 5), multichannel=True):
    # 转换为灰度图像
    obj1_gray = cv2.cvtColor(obj1, cv2.COLOR_RGB2GRAY)
    obj2_gray = cv2.cvtColor(obj2, cv2.COLOR_RGB2GRAY)
    # 计算SSIM
    return ssim(obj1_gray, obj2_gray, win_size=win_size, multichannel=False)


# 定义SSIM特征匹配函数
def match_objects(objects1, objects2, threshold=0.7):
    matched_results = []
    for obj1, class_id1 in objects1:
        matched = False
        for obj2, class_id2 in objects2:
            similarity = calculate_ssim(obj1, obj2)
            if similarity > threshold:
                matched = True
                break
        matched_results.append((matched, class_id1, class_id2))
    return matched_results

# 设定数据集路径
image_dir = 'images'
label_dir = 'labels'

# 加载数据集并获取目标区域图像
image_paths = sorted(os.listdir(image_dir))
label_paths = sorted(os.listdir(label_dir))
objects1 = crop_objects(os.path.join(image_dir, image_paths[0]), os.path.join(label_dir, label_paths[0]))
objects2 = crop_objects(os.path.join(image_dir, image_paths[1]), os.path.join(label_dir, label_paths[1]))

# 将两个图像resize到相同的尺寸
target_size = (256, 256)  # 定义目标尺寸
resized_objects1 = resize_objects(objects1, target_size)
resized_objects2 = resize_objects(objects2, target_size)

# 进行目标匹配并计算匹配成功率
matched_results = match_objects(resized_objects1, resized_objects2)
matching_accuracy = sum(result[0] for result in matched_results) / len(matched_results)

# 打印匹配的目标及其类别信息
print("Matching Accuracy:", matching_accuracy)
print("Matched Objects:")
for i, (matched, class_id1, class_id2) in enumerate(matched_results):
    if matched:
        print(f"Object {i + 1}: Class {class_id1} matched with Class {class_id2}")
