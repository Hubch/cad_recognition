import onnxruntime
from PIL import Image
import numpy as np
import logging
import LabelMap
import cv2
import utils

class YOLOv5ONNXPipeline:
    def __init__(self, 
                 onnx_model_path, 
                 img_size=640, 
                 conf_thres=0.25, 
                 iou_thres=0.45, 
                 stride=1,
                 line_thickness=2, 
                 text_thickness=1,
                 hide_labels=False, 
                 hide_conf=True,
                 half=False):
        # 初始化ONNX运行时会话
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.logger = logging.getLogger(__name__)
        
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

        print("input_shape:", self.session.get_inputs()[0].shape)


    def preprocess_image(self, image):
        # YOLOv5的预处理步骤通常包括：
        # 0. image: cv2 image with HWC and BGR
        # 1. 将图像调整到模型期望的尺寸
        # 2. 将PIL图像转换为NumPy数组
        # 3. 归一化像素值
        # 4. 增加一个批次维度
        # 注意：这里的尺寸和归一化参数需要根据模型进行调整
        image = utils.letterbox(image, new_shape=self.img_size)     # 使用填充进行 resize 避免失真
        image = image[:, :, ::-1].transpose(2, 0, 1)          # BGR -> RGB & HWC -> CHW
        if self.half:
            image = np.ascontiguousarray(image).astype(np.float16)
        else :
            image = np.ascontiguousarray(image).astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        assert len(image.shape) == 4
        self.logger.info(f"Image array shape: {image_array.shape}")
        return image
    
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
        
    def postprocess_results(self, boxes, classIds, confidences):
        # YOLOv5的后处理步骤通常包括：
        # 1. 解析模型输出，通常包括边界框坐标、置信度和类别
        # 2. 应用阈值来过滤低置信度的预测
        # 3. 应用非极大值抑制（NMS）来去除重叠的边界框
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, self.iou_thres)  # 执行nms算法
        pred_boxes = []
        pred_confes = []
        pred_classes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                confidence = confidences[i]
                if confidence >= self.threshold:
                    pred_boxes.append(boxes[i])
                    pred_confes.append(confidence)
                    pred_classes.append(classIds[i])
        return pred_boxes, pred_classes, pred_confes
    
    def draw_image_with_bbox(self, image, boxes, classIds, confidences):
        for i, _ in enumerate(boxes):
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            box = (left, top, left + width, top + height)
            box = np.squeeze(
                utils.scale_coords(self.img_size, np.expand_dims(box, axis=0).astype("float"), image.shape[1:]).round(), axis=0).astype(
                "int")  # 进行坐标还原
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            # 执行画图函数
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), thickness=self.line_thickness)
            cv2.putText(image, '{0}--{1:.2f}'.format(self.labels_map[classIds[i]], confidences[i]), (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=self.text_thickness)
        return image
        
    def __call__(self, image):
        # 预处理图像
        image = self.preprocess_image(image)
        # 推理
        boxes, classIds, confidences = self.detect_image(image)
        # 后处理
        boxes, classIds, confidences = self.postprocess_results(boxes, classIds, confidences)
        # 画图传回去
        image_with_box = self.draw_image_with_bbox(image, boxes, classIds, confidences)
        print ("boxes: {}, classIds: {}, confidence: {}".format(boxes, classIds, confidences))
        cv2.imwrite("./test.jpg", image_with_box)
        return boxes

