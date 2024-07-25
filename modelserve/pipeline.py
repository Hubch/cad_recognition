import onnxruntime
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import LabelMap
import cv2
from utils import utils
from utils.polt import Annotator,colors
import fitz  # PyMuPDF

class CADMatchingPipeline:
    def __init__(self, 
                 detect_model_path,
                 match_model_path,
                 threshold_path,
                 class_to_idx_path,
                 img_size=640, 
                 conf_thres=0.4, 
                 iou_thres=0.1, 
                 stride=1,
                 line_thickness=2, 
                 text_thickness=1,
                 hide_labels=False, 
                 hide_conf=True,
                 half=False,
                 threshold = 0.2):
        
        # Load the first ONNX model
        self.detect_session = onnxruntime.InferenceSession(detect_model_path)

        # Load the second ONNX model
        self.match_session = onnxruntime.InferenceSession(match_model_path)

        with open(class_to_idx_path, 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
        
        self.thre_euc, self.thre_cos = np.load(threshold_path)['arr_0']

        self.window_size = (640, 640)
        self.step_size = 480

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.stride = stride
        self.line_thickness = line_thickness
        self.text_thickness = text_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.labels_map = LabelMap.labels_map
        self.threshold = threshold
        self.ration = 0
        self.font = "msyh.ttc"
        self.colors = colors


    # pre-process procedure in below functions group
    def convert_pdf_to_images(self, pdf_path, dpi=300):
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):   # assert there are multi-pages included, though there is always one
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi / 72, dpi / 72)  
            pix = page.get_pixmap(matrix=mat)  
            img = Image.open(io.BytesIO(pix.tobytes()))
            images.append(img)
        return images
    
    def resize_to_same_height(self, image1, image2):
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
        
    
    def sliding_window(self, image, step_size, window_size):
        for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                # 添加填充，使窗口大小与window_size一致
                pad_bottom = max(0, window_size[1] - window.shape[0])
                pad_right = max(0, window_size[0] - window.shape[1])
                window = cv2.copyMakeBorder(window, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            yield (x, y, window)
    
    def cutting_image(self, image, step_size, window_size):
        windows = []
        for (x, y, window) in self.sliding_window(image, step_size, window_size):
            windows.append((x, y, window))
        return windows

    def cut_pdf_to_windows_array(self, pdf_paths):
        hd_images = []
        windows_batches = []
        array_xy = []
        for pdf_name in pdf_paths:
            hd_images.append(np.array(self.convert_pdf_to_images(pdf_name, dpi=200)[0])) # get the single pic from a list 
        
        hd_images[0], hd_images[1] = self.resize_to_same_height(hd_images[0], hd_images[1])

        for image in hd_images:
            # cut a huge image to a group of sub-images by sliding window 
            windows = self.cutting_image(image, self.step_size, self.window_size) 
            xy = [[x, y] for x, y, _ in windows]
            array_xy.append(xy)
        
            # compose the windowsed images to a batch tensor
            windows = [np.transpose(window, (2, 0, 1)).astype(np.float32)[np.newaxis, :] / 255.0 for _, _, window in windows]
            windows_array = np.concatenate(windows, axis=0)
            windows_batches.append(windows_array)
        
        return hd_images, windows_batches, array_xy


    # detect procedure in functions group below
    def postprocess_results(self, results, conf_threshold=0.25):
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

            all_boxes.append(np.array(boxes))
            all_scores.append(np.array(scores))

        return all_boxes, all_scores
    
    def integrate_detections(self, xy, detections, scores):
        global_boxes = []
        global_scores = []
        for (x, y), det, score in zip(xy, detections, scores):
            for box, s in zip(det, score):
                global_box = [box[0] + x, box[1] + y, box[2] + x, box[3] + y]
                global_boxes.append(global_box)
                global_scores.append(s)
        return np.array(global_boxes), np.array(global_scores)

    def global_nms(self, boxes, scores):
        if len(boxes) == 0:
            return []
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.iou_thres) 
        return indices
    
    def detect(self, array_windows, array_xy):
        bbox_results = []
        for array_window, xy in zip(array_windows, array_xy):
            results = self.detect_session.run(None, {"inputs": array_window})[0]
            all_boxes, all_scores = self.postprocess_results(results, self.conf_thres)

            # integraet local result of windows image to a global result  
            global_boxes, global_scores = self.integrate_detections(xy, all_boxes, all_scores)
            indices = self.global_nms(global_boxes, global_scores)
            final_boxes = global_boxes[indices]
            bbox_results.append(final_boxes)
        
        return bbox_results 


    # position matching procedure functions group below
    def normalize_bbox(self, image_shape, bbox):
        h, w = image_shape[:2]
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

    def calculate_distance(self, box1, box2):
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance

    def calculate_shape_similarity(self, box1, box2):
        width1, height1 = box1[2] - box1[0], box1[3] - box1[1]
        width2, height2 = box2[2] - box2[0], box2[3] - box2[1]
        shape_similarity = 1 - abs((width1 / height1) - (width2 / height2))
        return shape_similarity

    def calculate_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def calculate_center(self, box):
        """Calculate the center of a bounding box."""
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def calculate_relative_position_difference(self, box1, box2, image_shape1, image_shape2):
        """Calculate the relative position difference between two bounding boxes."""
        center1 = self.calculate_center(box1)
        center2 = self.calculate_center(box2)
        relative_center1 = [center1[0] / image_shape1[1], center1[1] / image_shape1[0]]
        relative_center2 = [center2[0] / image_shape2[1], center2[1] / image_shape2[0]]
        position_difference = np.linalg.norm(np.array(relative_center1) - np.array(relative_center2))
        return position_difference

    def match_optimized(self, boxes1, boxes2, image1, image2, distance_weight=1.0, shape_weight=1.0, 
                        area_weight=1.0, position_weight=1.0, position_threshold=0.25):
        matched_pairs = []

        norm_boxes1 = [self.normalize_bbox(image1.shape, box) for box in boxes1]
        norm_boxes2 = [self.normalize_bbox(image2.shape, box) for box in boxes2]

        cost_matrix = np.zeros((len(norm_boxes1), len(norm_boxes2)))

        for i, box1 in enumerate(norm_boxes1):
            for j, box2 in enumerate(norm_boxes2):
                distance = self.calculate_distance(box1, box2)
                shape_similarity = self.calculate_shape_similarity(box1, box2)
                area1 = self.calculate_area(box1)
                area2 = self.calculate_area(box2)
                area_ratio = min(area1, area2) / max(area1, area2)
                position_difference = self.calculate_relative_position_difference(box1, box2, image1.shape, image2.shape)

                cost = distance_weight * distance - shape_weight * shape_similarity - area_weight * area_ratio + position_weight * position_difference
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for idx, (r, c) in enumerate(zip(row_ind, col_ind)):
            position_difference = self.calculate_relative_position_difference(boxes1[r], boxes2[c], image1.shape, image2.shape)
            if cost_matrix[r, c] < 1.0 and position_difference < position_threshold:
                matched_pairs.append((r, c))

        return matched_pairs
    

    # image matching procedure functions group below
    def crop_and_preprocess(self, image, bbox, target_size=(32, 32)):
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        array = gray[np.newaxis, np.newaxis, :] / 255.0
        return array

    def calculate_cosine_distance(self, feat1, feat2):
        dot_product = np.sum(feat1 * feat2, axis=1)
        norm_feat1 = np.linalg.norm(feat1, axis=1)
        norm_feat2 = np.linalg.norm(feat2, axis=1)
        cosine_similarity = dot_product / (norm_feat1 * norm_feat2)
        return 1 - cosine_similarity

    def match_images(self, image1, image2, boxes1, boxes2, matches, threshold=0.5):
        same_entity_count = 0
        total_matches = len(matches)
        img_match_success = 0
        img_match_fail = 0

        # 提取所有匹配的裁剪图像并批量处理
        arraies1 = []
        arraies2 = []
        bboxes1 = []
        bboxes2 = []

        for (i, j) in matches:
            array1 = self.crop_and_preprocess(image1, boxes1[i])
            array2 = self.crop_and_preprocess(image2, boxes2[j])
            arraies1.append(array1)
            arraies2.append(array2)
            bboxes1.append(boxes1[i])
            bboxes2.append(boxes2[j])

        # 将所有图像批量化
        batch1 = np.concatenate(arraies1, axis=0)
        batch2 = np.concatenate(arraies2, axis=0)

        matched_results = []

        feats1, logits1 = self.match_session.run(["feature", "logits"], {"input": batch1})
        feats2, logits2 = self.match_session.run(["feature", "logits"], {"input": batch2})
        clses1 = np.argmax(logits1, axis=1)
        clses2 = np.argmax(logits2, axis=1)


        for feat1, feat2, cls1, cls2, bbox1, bbox2 in zip(feats1, feats2, clses1, clses2, bboxes1, bboxes2):
            cosine_distance = self.calculate_cosine_distance(np.expand_dims(feat1, axis=0), np.expand_dims(feat2, axis=0))
            bbox1 = [float(x) for x in bbox1]
            bbox2 = [float(x) for x in bbox2]
            matched_results.append([float(cosine_distance), int(cls1), int(cls2), bbox1, bbox2])

            if cosine_distance < threshold:
                same_entity_count += 1
                img_match_success += 1
            else:
                img_match_fail += 1

        return same_entity_count, total_matches, matched_results, img_match_success, img_match_fail
    
    def judge_if_same_entity(self, same_entity_count, total_matches, confidence_threshold=0.75):
        if total_matches == 0:
            return False
        match_ratio = same_entity_count / total_matches
        return match_ratio >= confidence_threshold
    

    # drawing and json generation functions group is below
    def draw_bboxes(self, image, boxes, color=(0, 255, 0)):
        for box in boxes:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    def visualize_matches_with_info(self, image1, image2, boxes1, boxes2, matches, match_results, threshold, 
                                bbox_match_success, bbox_match_fail, img_match_success, img_match_fail):
        
        combined_image = np.concatenate((image1, image2), axis=1)
        offset = image1.shape[1]

        self.draw_bboxes(image1, boxes1, color=(0, 255, 0))
        self.draw_bboxes(image2, boxes2, color=(255, 0, 0))
        
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
                "is_pair": cosine < self.thre_cos,  # cls1 == cls2
                "cosine_distance": cosine
            } for cosine, cls1, cls2, bbox1, bbox2 in match_results
            ]
        }

        return result_json

    
        
    def __call__(self, pdf_paths):
        # get high resolution images  
        hd_images, array_windows, array_xy = self.cut_pdf_to_windows_array(pdf_paths)
        bbox_results = self.detect(array_windows, array_xy)
        
        # match image
        matches = self.match_optimized(bbox_results[0], bbox_results[1], hd_images[0], hd_images[1])

        same_entity_count, total_matches, matched_results, img_match_success, img_match_fail = \
            self.match_images(hd_images[0], hd_images[1], bbox_results[0], bbox_results[1], matches, self.thre_cos)
        
        is_same_entity = self.judge_if_same_entity(same_entity_count, total_matches)

        bbox_match_success = len(matches)
        bbox_match_fail = len(bbox_results[0]) + len(bbox_results[1]) - 2 * len(matches)

        return self.visualize_matches_with_info(hd_images[0], hd_images[1], bbox_results[0], bbox_results[1], 
                                                matches, matched_results, threshold=self.thre_cos,
                                                bbox_match_success=bbox_match_success, bbox_match_fail=bbox_match_fail, 
                                                img_match_success=img_match_success, img_match_fail=img_match_fail)




class OCRPipeline:
    def __init__(self,use_angle_cls=True, lang="ch",use_gpu=True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang,use_gpu=use_gpu)  # 这里设置为中文识    

    def preprocess_image(self,image):
        return image

    def postprocess_results(self,original_image,result):

        draw_image = self.draw_text_on_white_image(original_image,result)
        # 准备JSON格式的数据结构
        ocr_results = {
            "draw_image":utils.convertBase64(draw_image),
            "detections": [],
            "texts":[line[1][0] for line in result]
        }

        # 遍历PaddleOCR的输出结果
        for line in result:
            # line格式为[[[x_min, y_min, x_max, y_max],(text, confidence)]]
            ocr_results["detections"].append({
                "bbox": [[line[0]][0][0],[line[0]][0][1],[line[0]][0][2],[line[0]][0][3]], # 坐标
                "text": [line[1]][0][0],  # 文本内容
                "confidence": [line[1]][0][1]  # 置信度
            })
        return ocr_results

    def draw_text_on_white_image(self,image,result, font_path='simsun.ttc'):
        # 准备绘制参数
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        return im_show
    
 
    
    def __call__(self, image):
        image_deal = self.preprocess_image(image)
        results = self.ocr.ocr(image_deal)
        detections= self.postprocess_results(image,results[0])
        return detections


    