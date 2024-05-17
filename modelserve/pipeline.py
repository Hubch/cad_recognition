import onnxruntime
from PIL import Image
import numpy as np
import logging

class Detector():

    def __init__(self, opt):
        super(Detector, self).__init__()
        self.img_size = opt.img_size
        self.threshold = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.stride = 1
        self.weights = opt.weights
        self.init_model()
        self.names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                      "traffic light",
                      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                      "cow",
                      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
                      "frisbee",
                      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                      "surfboard",
                      "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                      "apple",
                      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                      "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear",
                      "hair drier", "toothbrush"]

    def init_model(self):
        """
        模型初始化这一步比较固定写法
        :return:
        """
        sess = onnxruntime.InferenceSession(self.weights)  # 加载模型权重
        self.input_name = sess.get_inputs()[0].name  # 获得输入节点
        output_names = []
        for i in range(len(sess.get_outputs())):
            print("output node:", sess.get_outputs()[i].name)
            output_names.append(sess.get_outputs()[i].name)  # 所有的输出节点
        print(output_names)
        self.output_name = sess.get_outputs()[0].name  # 获得输出节点的名称
        print(f"input name {self.input_name}-----output_name{self.output_name}")
        input_shape = sess.get_inputs()[0].shape  # 输入节点形状
        print("input_shape:", input_shape)
        self.m = sess

    def preprocess(self, img):
        """
        图片预处理过程
        :param img:
        :return:
        """
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]  # 图片预处理
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        assert len(img.shape) == 4
        return img0, img

    def detect(self, im):
        """

        :param img:
        :return:
        """
        img0, img = self.preprocess(im)
        pred = self.m.run(None, {self.input_name: img})[0]  # 执行推理
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
        return im, pred_boxes, pred_confes, pred_classes


def main(opt):
    det = Detector(opt)
    image = cv2.imread(opt.img)
    shape = (det.img_size, det.img_size)

    img, pred_boxes, pred_confes, pred_classes = det.detect(image)
    if len(pred_boxes) > 0:
        for i, _ in enumerate(pred_boxes):
            box = pred_boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            box = (left, top, left + width, top + height)
            box = np.squeeze(
                scale_coords(shape, np.expand_dims(box, axis=0).astype("float"), img.shape[:2]).round(), axis=0).astype(
                "int")  # 进行坐标还原
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            # 执行画图函数
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), thickness=2)
            cv2.putText(image, '{0}--{1:.2f}'.format(pred_classes[i], pred_confes[i]), (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
    cv2.imshow("detector", image)
    cv2.waitKey(0)


class YOLOv5ONNXPipeline:
    def __init__(self, 
                 onnx_model_path, 
                 img_size=640, 
                 conf_thres=0.25, 
                 iou_thres=0.45, 
                 stride=1
                 line_thickness=1, 
                 hide_labels=False, 
                 hide_conf=True):
        # 初始化ONNX运行时会话
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.logger = logging.getLogger(__name__)
        
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.stride = stride
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf


    def preprocess_image(self, image):
        # YOLOv5的预处理步骤通常包括：
        # 1. 将图像调整到模型期望的尺寸
        # 2. 将PIL图像转换为NumPy数组
        # 3. 归一化像素值
        # 4. 增加一个批次维度
        # 注意：这里的尺寸和归一化参数需要根据模型进行调整
        image = image.resize((640, 640))
        image_array = np.array(image).astype(np.float32)
        image_array = np.transpose(image_array, [2, 0, 1])  # 转换为CHW格式
        image_array /= 255.0  # 归一化
        image_array = np.expand_dims(image_array, axis=0)  # 增加批次维度
        self.logger.info(f"Image array shape: {image_array.shape}")
        return image_array

    def postprocess_results(self, results):
        # YOLOv5的后处理步骤通常包括：
        # 1. 解析模型输出，通常包括边界框坐标、置信度和类别
        # 2. 应用阈值来过滤低置信度的预测
        # 3. 应用非极大值抑制（NMS）来去除重叠的边界框
        return results

    def __call__(self, image):
        # 预处理图像
        image_array = self.preprocess_image(image)
        # 推理
        results = self.session.run(None, {self.input_name: image_array})
        # 后处理
        detections = self.postprocess_results(results[0])

        return detections


class YOLOv8SegmentationPipeline:
    def __init__(self, onnx_model_path):
        # 初始化ONNX模型
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_image(self, image):
        # YOLOv8的预处理步骤可能包括：
        # 1. 将图像调整到模型期望的尺寸
        # 2. 将PIL图像转换为NumPy数组
        # 3. 归一化像素值
        # 4. 增加一个批次维度
        # 注意：这里的尺寸和归一化参数需要根据你的YOLOv8模型进行调整

        # 假设模型期望的输入尺寸是640X640
        image = image.resize((640, 640))
        image_array = np.array(image).astype(np.float32)
        image_array /= 255.0  # 归一化
        image_array = np.expand_dims(image_array, axis=0)  # 增加批次维度
        
        return image_array

    def postprocess_results(self, results):
        # YOLOv8的后处理步骤可能包括：
        # 1. 解析模型输出，通常包括边界框坐标、置信度和类别
        # 2. 应用阈值来过滤低置信度的预测
        # 3. 应用非极大值抑制（NMS）来去除重叠的边界框
        # 注意：这里的后处理逻辑需要根据你的YOLOv8模型输出进行调整
        # 这里只是一个示例，实际的后处理代码会更复杂

        # 假设results是模型输出的直接结果，包含了分割掩码
        # 这里我们直接返回results，实际使用时需要根据模型输出格式进行后处理
        return results

    def __call__(self, image):
        # 预处理图像
        image_array = self.preprocess_image(image)
        # 推理
        results = self.session.run(None, {self.input_name: image_array})
        # 后处理
        segmentation = self.postprocess_results(results)

        return segmentation
