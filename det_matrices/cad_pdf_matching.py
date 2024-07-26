import fitz  # PyMuPDF
from PIL import Image
import io
import os 
import cv2
import numpy as np
import torch
import json
from scipy.optimize import linear_sum_assignment

from _utils.models import get_student_model


def load_det_model(model_path):
    import os 
    import sys
    yolov5_path = os.path.join(os.getcwd(), 'yolov5')
    if yolov5_path not in sys.path:
        sys.path.append(yolov5_path)

    from yolov5.models.common import DetectMultiBackend
    model = DetectMultiBackend(model_path)
    return model

def load_class_mapping(class_map_path):
    with open(class_map_path, 'r') as f:
        return json.load(f)

def convert_pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)  
        pix = page.get_pixmap(matrix=mat)  
        img = Image.open(io.BytesIO(pix.tobytes()))
        images.append(img)
    return images

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

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                # 添加填充，使窗口大小与window_size一致
                pad_bottom = max(0, window_size[1] - window.shape[0])
                pad_right = max(0, window_size[0] - window.shape[1])
                window = cv2.copyMakeBorder(window, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            yield (x, y, window)

def process_image(image, step_size, window_size):
    windows = []
    for (x, y, window) in sliding_window(image, step_size, window_size):
        windows.append((x, y, window))
    return windows

def detect_objects(model, input_tensor_batch):
    results = model(input_tensor_batch)
    return results

def postprocess_results(results, conf_threshold=0.45):
    all_boxes = []
    all_scores = []
    for result in results:
        pred = result
        pred = pred[pred[:, 4] > conf_threshold]

        if pred.shape[0] == 0:
            all_boxes.append(np.array([]))
            all_scores.append(np.array([]))
            continue

        boxes = pred[:, :4]
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        scores = pred[:, 4]

        all_boxes.append(boxes.cpu().numpy())
        all_scores.append(scores.cpu().numpy())

    return all_boxes, all_scores


def integrate_detections(windows, detections, scores):
    global_boxes = []
    global_scores = []
    for (x, y, _), det, score in zip(windows, detections, scores):
        for box, s in zip(det, score):
            global_box = [box[0] + x, box[1] + y, box[2] + x, box[3] + y]
            global_boxes.append(global_box)
            global_scores.append(s)
    return np.array(global_boxes), np.array(global_scores)

def global_nms(boxes, scores, iou_threshold=0.1):
    if len(boxes) == 0:
        return []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    return indices.numpy()


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


def crop_and_preprocess(image, bbox, target_size=(32, 32)):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

def calculate_cosine_distance(feat1, feat2):
    return 1 - torch.nn.functional.cosine_similarity(feat1, feat2).item()

def match_images(image1, image2, boxes1, boxes2, matches, match_model, threshold=0.5):
    match_model.eval()
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

    matched_results = []

    with torch.no_grad():
        feats1, logits1 = match_model(batch1)
        feats2, logits2 = match_model(batch2)
        clses1 = torch.argmax(logits1, dim=1).cpu().numpy()
        clses2 = torch.argmax(logits2, dim=1).cpu().numpy()

    for feat1, feat2, cls1, cls2, bbox1, bbox2 in zip(feats1, feats2, clses1, clses2, bboxes1, bboxes2):
        cosine_distance = calculate_cosine_distance(feat1.unsqueeze(0), feat2.unsqueeze(0))
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]
        matched_results.append([float(cosine_distance), int(cls1), int(cls2), bbox1, bbox2])

        if cosine_distance < threshold:
            same_entity_count += 1
            img_match_success += 1
        else:
            img_match_fail += 1

    return same_entity_count, total_matches, matched_results, img_match_success, img_match_fail


def judge_if_same_entity(same_entity_count, total_matches, confidence_threshold=0.6):
    if total_matches == 0:
        return False
    match_ratio = same_entity_count / total_matches
    return match_ratio >= confidence_threshold


def draw_bboxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

def draw_lines_with_info(image, boxes1, boxes2, matches, results, threshold, offset, image1_shape, image2_shape):
    for idx, ((i, j), result) in enumerate(zip(matches, results)):
        distance = result[0]
        print(f'distance: {distance}')
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

def visualize_matches_with_info(image1, image2, boxes1, boxes2, matches, match_results, threshold, 
                                bbox_match_success, bbox_match_fail, img_match_success, img_match_fail,
                                output='test'):
    '''
    result_json: {
    item_pairs: [
        {
            type1: '',
            type2: '',
            bbox1: '',
            bbox2: '',
            is_pair: True,
            cosine_distance: int
        },
    ]}
    '''
    json_path = os.path.join('./outputs', f'{output}_result.json')
    matched_img_path = os.path.join('./outputs', f'{output}_matched.png')

    combined_image = np.concatenate((image1, image2), axis=1)
    offset = image1.shape[1]

    draw_bboxes(image1, boxes1, color=(0, 255, 0))
    draw_bboxes(image2, boxes2, color=(255, 0, 0))

    combined_image_with_bboxes = np.concatenate((image1, image2), axis=1)

    draw_lines_with_info(combined_image_with_bboxes, boxes1, boxes2, matches, match_results, threshold, offset, image1.shape, image2.shape)

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

    cv2.imwrite(matched_img_path, final_image)
    print(f"Matched results saved to {matched_img_path}")

    
    result_json = {
        "global_info": {
            "Image 1 boxes": len(boxes1),
            "Image 2 boxes": len(boxes2),
            "BBox match success": bbox_match_success,
            "BBox match fail": bbox_match_fail,
            "Image match success": img_match_success,
            "Image match fail": img_match_fail,
            "Cosine distance threshold": threshold
        },
        "item_pairs": [
        {
            "type1": cls1,
            "type2": cls2,
            "bbox1": bbox1,
            "bbox2": bbox2,
            "is_pair": cosine < 0.3,  # for example
            "cosine_distance": cosine
        } for cosine, cls1, cls2, bbox1, bbox2 in match_results
        ]
    }
    print(result_json)
    with open(json_path, 'w') as json_file:
        json.dump(result_json, json_file, indent=4)
    print(f"Matched result json saved to {json_path}")

    return result_json
    



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdf_paths = ['02A', '02B']
    det_model_path = './weights/cad_yolo.pt'
    match_model_path = './weights/cad_hybrid'
    threshold_path = './weights/cad_threshold_student.npz'
    class_to_idx_path = './weights/class_map.json'

    class_to_idx = load_class_mapping(class_to_idx_path)
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    window_size = (640, 640)
    step_size = 480

    detection_model = load_det_model(det_model_path)
    match_model = get_student_model(num_classes=len(idx_to_class), 
                                    dropout_p=0, 
                                    init=False, 
                                    update=False, 
                                    weight_path=match_model_path, 
                                    cuda=True if torch.cuda.is_available() else False).to(device)
    
    
    hd_images = []
    bbox_results = []
    for pdf_name in pdf_paths:
        pdf_path = f'./data/{pdf_name}.pdf'
        # assert one page pre pdf file
        hd_images.append(np.array(convert_pdf_to_images(pdf_path, dpi=200)[0]))
    
    # aligned in the shape
    hd_images[0], hd_images[1] = resize_to_same_height(hd_images[0], hd_images[1])
    print(f'resized shape: {hd_images[0].shape}, {hd_images[1].shape}')
    
    for image in hd_images:
       
        windows = process_image(image, step_size, window_size)
        
        # compose the windowsed images to a batch tensor
        window_tensors = [torch.from_numpy(window).permute(2, 0, 1).float().unsqueeze(0) / 255.0 for _, _, window in windows]
        input_tensor_batch = torch.cat(window_tensors, dim=0)
        
        results = detect_objects(detection_model, input_tensor_batch)[0]        
        all_boxes, all_scores = postprocess_results(results)
        
        # integraet local result of windows image to a global result  
        global_boxes, global_scores = integrate_detections(windows, all_boxes, all_scores)
        indices = global_nms(global_boxes, global_scores)
        final_boxes = global_boxes[indices]
        bbox_results.append(final_boxes)

        draw_bboxes(image, final_boxes)
        
        result_image = Image.fromarray(image)
        result_image.save(f'./outputs/{pdf_name}_detected.png')
    
    matches = match_optimized(bbox_results[0], bbox_results[1], hd_images[0], hd_images[1])
    thre_euc, thre_cos = np.load(threshold_path)['arr_0']
    same_entity_count, total_matches, matched_results, img_match_success, img_match_fail = \
            match_images(hd_images[0], hd_images[1], bbox_results[0], bbox_results[1], \
                         matches, match_model, thre_cos)
    # to judge if the both matched according a threshold
    is_same_entity = judge_if_same_entity(same_entity_count, total_matches)

    bbox_match_success = len(matches)
    bbox_match_fail = len(bbox_results[0]) + len(bbox_results[1]) - 2 * len(matches)


    # visualization 
    print(f'matched_results: {matched_results}')
    json_result = visualize_matches_with_info(
                        hd_images[0], hd_images[1], bbox_results[0], bbox_results[1], matches, matched_results, 
                        threshold=thre_cos, bbox_match_success=bbox_match_success, bbox_match_fail=bbox_match_fail, 
                        img_match_success=img_match_success, img_match_fail=img_match_fail, output='test')
    

if __name__ == "__main__":
    main()