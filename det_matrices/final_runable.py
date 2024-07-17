import torch
import cv2
import numpy as np
import random
import os
from scipy.optimize import linear_sum_assignment

from utils.models import get_model

def load_det_model(model_path):
    from yolov5.models.common import DetectMultiBackend
    model = DetectMultiBackend(model_path)
    return model

def read_images(image_paths):
    return [cv2.imread(img_path) for img_path in image_paths]

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)

def resize_to_same_height(image1, image2):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    if h1 > h2:
        new_w2 = int(w2 * (h1 / h2))
        image2_resized = cv2.resize(image2, (new_w2, h1))
        return image1, image2_resized
    else:
        new_w1 = int(w1 * (h2 / h1))
        image1_resized = cv2.resize(image1, (new_w1, h2))
        return image1_resized, image2

def preprocess_images(image_paths, new_shape=(640, 640), color=(114, 114, 114)):
    images = [cv2.imread(img_path) for img_path in image_paths]
    
    image1, image2 = resize_to_same_height(images[0], images[1])

    input_images = []
    ratios = []
    pads = []

    for img in [image1, image2]:
        img_resized, ratio, pad = letterbox(img, new_shape, color)
        input_images.append(img_resized)
        ratios.append(ratio)
        pads.append(pad)

    input_tensors = [torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0 for img_resized in input_images]
    input_batch = torch.cat(input_tensors, 0)
    
    return input_batch, ratios, pads, [image1, image2]

def detect_objects(model, input_batch, conf_threshold=0.45, iou_threshold=0.1):
    results = model(input_batch)
    return results

def postprocess_results(results, images, ratios, pads, original_shapes, conf_threshold=0.45, iou_threshold=0.1):
    nms_results = []
    for i, (pred, ratio, pad, original_shape) in enumerate(zip(results, ratios, pads, original_shapes)):
        pred = pred[pred[:, 4] > conf_threshold]  # 过滤低置信度框

        if pred.shape[0] == 0:
            nms_results.append(np.array([]))
            continue

        boxes = pred[:, :4]
        boxes[:,0] -= boxes[:, 2] / 2
        boxes[:,1] -= boxes[:, 3] / 2
        boxes[:,2] += boxes[:, 0]
        boxes[:,3] += boxes[:, 1]
        scores = pred[:, 4]

        indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold=iou_threshold)
        boxes = boxes[indices]

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, [0, 2]] /= ratio  # x ratio
        boxes[:, [1, 3]] /= ratio  # y ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])  # 确保边界框在图像边界内
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])

        nms_results.append(boxes.cpu().numpy())

    return nms_results

def normalize_bbox(image_shape, bbox):
    h, w = image_shape[:2]
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def calculate_shape_similarity(box1, box2):
    width1, height1 = box1[2] - box1[0], box1[3] - box1[1]
    width2, height2 = box2[2] - box2[0], box2[3] - box2[1]
    shape_similarity = 1 - abs((width1 / height1) - (width2 / height2))
    return shape_similarity

def calculate_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def calculate_center(box):
    """Calculate the center of a bounding box."""
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

def calculate_relative_position_difference(box1, box2, image_shape1, image_shape2):
    """Calculate the relative position difference between two bounding boxes."""
    center1 = calculate_center(box1)
    center2 = calculate_center(box2)
    relative_center1 = [center1[0] / image_shape1[1], center1[1] / image_shape1[0]]
    relative_center2 = [center2[0] / image_shape2[1], center2[1] / image_shape2[0]]
    position_difference = np.linalg.norm(np.array(relative_center1) - np.array(relative_center2))
    return position_difference

def match_optimized(boxes1, boxes2, image1, image2, distance_weight=1.0, shape_weight=1.0, area_weight=1.0, position_weight=1.0, position_threshold=0.25):
    matched_pairs = []

    norm_boxes1 = [normalize_bbox(image1.shape, box) for box in boxes1]
    norm_boxes2 = [normalize_bbox(image2.shape, box) for box in boxes2]

    cost_matrix = np.zeros((len(norm_boxes1), len(norm_boxes2)))

    for i, box1 in enumerate(norm_boxes1):
        for j, box2 in enumerate(norm_boxes2):
            distance = calculate_distance(box1, box2)
            shape_similarity = calculate_shape_similarity(box1, box2)
            area1 = calculate_area(box1)
            area2 = calculate_area(box2)
            area_ratio = min(area1, area2) / max(area1, area2)
            position_difference = calculate_relative_position_difference(box1, box2, image1.shape, image2.shape)

            cost = distance_weight * distance - shape_weight * shape_similarity - area_weight * area_ratio + position_weight * position_difference
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for idx, (r, c) in enumerate(zip(row_ind, col_ind)):
        position_difference = calculate_relative_position_difference(boxes1[r], boxes2[c], image1.shape, image2.shape)
        if cost_matrix[r, c] < 1.0 and position_difference < position_threshold:
            matched_pairs.append((r, c))

    return matched_pairs


def draw_bboxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

def draw_lines_with_info(image, boxes1, boxes2, matches, cosine_distances, threshold, offset, image1_shape, image2_shape):
    for idx, ((i, j), distance) in enumerate(zip(matches, cosine_distances)):
        color = (0, 100, 0) if distance < threshold else (0, 0, 255)  # 深绿色或红色
        box1 = boxes1[i]
        box2 = boxes2[j]
        center1 = (int((box1[0] + box1[2]) / 2), int((box1[1] + box1[3]) / 2))
        center2 = (int((box2[0] + box2[2]) / 2 + offset), int((box2[1] + box2[3]) / 2))

        # 计算相对位置差异
        position_difference = calculate_relative_position_difference(box1, box2, image1_shape, image2_shape)

        cv2.line(image, center1, center2, color, 2)
        cv2.putText(image, f'{distance:.2f}', (center1[0] + 5, center1[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
        cv2.putText(image, f'{position_difference:.2f}', (center2[0] + 5, center2[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

def add_info_text(image, info_text, font_scale=2, color=(0, 0, 0), thickness=3):
    line_height = int(font_scale * 20 * 2)  # 调整行距为 2 倍
    for i, line in enumerate(info_text.split('\n')):
        y = (i + 1) * line_height
        cv2.putText(image, line, (100, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def visualize_matches_with_info(image1, image2, boxes1, boxes2, matches, cosine_distances, threshold, 
                                bbox_match_success, bbox_match_fail, img_match_success, img_match_fail,
                                output_path='matched_results_with_info.png'):
    combined_image = np.concatenate((image1, image2), axis=1)
    offset = image1.shape[1]

    draw_bboxes(image1, boxes1, color=(0, 255, 0))
    draw_bboxes(image2, boxes2, color=(255, 0, 0))

    combined_image_with_bboxes = np.concatenate((image1, image2), axis=1)

    draw_lines_with_info(combined_image_with_bboxes, boxes1, boxes2, matches, cosine_distances, threshold, offset, image1.shape, image2.shape)

    # 创建一个新的空白图像，用于显示文本信息
    text_area_height = 250  # 可以根据需要调整高度
    text_image = np.ones((text_area_height, combined_image_with_bboxes.shape[1], 3), dtype=np.uint8) * 255

    # 添加说明信息
    info_text = (f"Image 1 boxes: {len(boxes1)}     "
                 f"Image 2 boxes: {len(boxes2)}\n"
                 f"BBox match success: {bbox_match_success}    "
                 f"BBox match fail: {bbox_match_fail}\n"
                 f"Image match success: {img_match_success}    "
                 f"Image match fail: {img_match_fail}\n"
                 f"Cosine distance threshold: {threshold}")
    add_info_text(text_image, info_text)

    # 将文本区域与主要图像区域组合
    final_image = np.concatenate((text_image, combined_image_with_bboxes), axis=0)

    cv2.imwrite(output_path, final_image)
    print(f"Matched results saved to {output_path}")


def crop_and_preprocess(image, bbox, target_size=(32, 32)):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

def calculate_cosine_distance(feat1, feat2):
    return 1 - torch.nn.functional.cosine_similarity(feat1, feat2).item()


def match_images(image1, image2, boxes1, boxes2, matches, feature_model, threshold=0.5):
    feature_model.eval()
    same_entity_count = 0
    total_matches = len(matches)
    img_match_success = 0
    img_match_fail = 0

    # 提取所有匹配的裁剪图像并批量处理
    tensors1 = []
    tensors2 = []
    bboxes1 = []
    bboxes2 = []

    for (i, j) in matches:
        tensor1 = crop_and_preprocess(image1, boxes1[i])
        tensor2 = crop_and_preprocess(image2, boxes2[j])
        tensors1.append(tensor1)
        tensors2.append(tensor2)
        bboxes1.append(boxes1[i])
        bboxes2.append(boxes2[j])

    # 将所有图像批量化
    batch1 = torch.cat(tensors1, dim=0)
    batch2 = torch.cat(tensors2, dim=0)

    cosine_distances = []

    with torch.no_grad():
        feats1 = feature_model(batch1)
        feats2 = feature_model(batch2)

    for feat1, feat2, bbox1, bbox2 in zip(feats1, feats2, bboxes1, bboxes2):
        cosine_distance = calculate_cosine_distance(feat1.unsqueeze(0), feat2.unsqueeze(0))
        cosine_distances.append(cosine_distance)
        if cosine_distance < threshold:
            same_entity_count += 1
            img_match_success += 1
        else:
            img_match_fail += 1

    return same_entity_count, total_matches, cosine_distances, img_match_success, img_match_fail

def judge_if_same_entity(same_entity_count, total_matches, confidence_threshold=0.8):
    if total_matches == 0:
        return False
    match_ratio = same_entity_count / total_matches
    return match_ratio >= confidence_threshold

def main():
    det_model_path = './weights/cad_yolo.pt'
    match_model_path = './weights/cad_matrice.pth'
    image_paths = ['img1.png', 'img2.png']
    threshold_path = './weights/cad_thres.npz'
    
    # 载入模型
    detection_model = load_det_model(det_model_path)
    feature_model = get_model(init=False, update=False, cuda=False, weight_path=match_model_path)
    
    # 预处理图像
    input_batch, ratios, pads, resized_images = preprocess_images(image_paths)
    
    # 目标检测
    results = detect_objects(detection_model, input_batch)[0]
    nms_results = postprocess_results(results, resized_images, ratios, pads, [img.shape for img in resized_images])
    
    # 目标匹配
    matches = match_optimized(nms_results[0], nms_results[1], resized_images[0], resized_images[1])
    
    # 图像匹配
    thre_euc, thre_cos = np.load(threshold_path)['arr_0']
    same_entity_count, total_matches, cosine_distances, img_match_success, img_match_fail = match_images(resized_images[0], resized_images[1], nms_results[0], nms_results[1], matches, feature_model, thre_cos)
    
    # 综合判断
    is_same_entity = judge_if_same_entity(same_entity_count, total_matches)
    
    # 统计 bbox 匹配结果
    bbox_match_success = len(matches)
    bbox_match_fail = len(nms_results[0]) + len(nms_results[1]) - 2 * len(matches)

    # 可视化匹配结果并添加匹配信息
    visualize_matches_with_info(resized_images[0], resized_images[1], nms_results[0], nms_results[1], matches, cosine_distances, threshold=thre_cos, 
                                bbox_match_success=bbox_match_success, bbox_match_fail=bbox_match_fail, 
                                img_match_success=img_match_success, img_match_fail=img_match_fail)
    
    print(f"Are the two CAD images describing the same entity? {'Yes' if is_same_entity else 'No'}")

if __name__ == "__main__":
    main()