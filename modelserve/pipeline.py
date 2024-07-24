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
        self.detect_input_name = self.detect_session.get_inputs()[0].name
        self.detect_output_name = self.detect_session.get_outputs()[0].name

        # Load the second ONNX model
        self.match_session = onnxruntime.InferenceSession(match_model_path)
        self.match_input_name = self.match_session.get_inputs()[0].name
        self.match_output_feature = self.match_session.get_outputs()[0].name
        self.match_output_logits = self.match_session.get_outputs()[1].name

        with open(class_to_idx_path, 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
        self.threshold_path = threshold_path
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
    
    def process_image(self, image, step_size, window_size):
        windows = []
        for (x, y, window) in self.sliding_window(image, step_size, window_size):
            windows.append((x, y, window))
        return windows

    def preprocess_image(self, pdf_paths):
        hd_images = []
        for pdf_name in pdf_paths:
            hd_images.append(np.array(self.convert_pdf_to_images(pdf_name, dpi=200)[0])) # get the single pic from a list 
        
        hd_images[0], hd_images[1] = self.resize_to_same_height(hd_images[0], hd_images[1])

        for image in hd_images:
            # cut a huge image to a group of sub-images by sliding window 
            windows = self.process_image(image, self.step_size, self.window_size) 
        
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

    
    def detect_image(self, image):
        pred = self.session.run(None, {self.input_name: image})[0]  # 执行推理
        pred = pred.astype(np.float32)
        pred = np.squeeze(pred, axis=0)
        boxes = []
        classIds = []
        confidences = []
        for detection in pred:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID] * detection[4]  # 置信度为类别的概率和目标框概率值得乘积

            if confidence > self.threshold:
                box = detection[0:4]
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))

        return boxes, classIds, confidences
        
    def postprocess_results(self,image, boxes, classIds, confidences):
        # YOLOv5的后处理步骤通常包括：
        # 1. 解析模型输出，通常包括边界框坐标、置信度和类别
        # 2. 应用阈值来过滤低置信度的预测
        # 3. 应用非极大值抑制（NMS）来去除重叠的边界框
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, self.iou_thres)  # 执行nms算法
        pred_boxes = []
        pred_confes = []
        pred_classes = []
        pred_texts = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                confidence = confidences[i]
                if confidence >= self.threshold:
                    # 进行坐标还原
                    box = boxes[i]
                    left, top, right, bottom = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                    box = [left, top, right, bottom]
                    box = np.squeeze(
                        utils.scale_coords(self.img_size, np.expand_dims(box, axis=0).astype("float"), image.shape[:2]).round(), axis=0).astype(
                        "int").tolist()
                    pred_boxes.append(box)
                    pred_confes.append(confidence)
                    pred_classes.append(classIds[i])
                    pred_texts.append(self.labels_map[classIds[i]])
        return pred_boxes, pred_classes, pred_confes,pred_texts
    
    def draw_image_with_bbox(self, image, boxes, classIds, confidences):
        annotator = Annotator(image, example=str("闸阀"),font_size=12,font=self.font,pil=True)
        for i, _ in enumerate(boxes):
            box = boxes[i]
            # left, top, right, bottom = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            # 调试输出，检查原始边界框坐标
            className = self.labels_map.get(classIds[i], None)
            if className is not None :
                color = self.colors(classIds[i], True)
                annotator.box_label(box= box, label= className, color=color,rotated = False)
                
        im0 = annotator.result()
        cv2.imwrite("run/output.jpg", im0)
        return im0
        
    def __call__(self, pdf_paths):
        # 预处理图像
        image_deal = self.preprocess_image(pdf_paths)
        # 推理
        boxes, classIds, confidences = self.detect_image(image_deal)
        # 后处理
        boxes, classIds, confidences,texts = self.postprocess_results(image,boxes, classIds, confidences)
        # 画图传回去
        image_with_box = self.draw_image_with_bbox(image, boxes, classIds, confidences)
        image_base64_str = utils.convertBase64(image_with_box)
        result = {
            "image_with_box":image_base64_str,
            "boxes":boxes, 
            "classIds":classIds, 
            "classtexts":texts, 
            "confidences": [round(num, 2) for num in confidences]
            }
        return result


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


    