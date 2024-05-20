import onnxruntime
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import json
from io import BytesIO
import base64
import cv2

class YOLOv5ONNXPipeline:
    def __init__(self, onnx_model_path):
        # 初始化ONNX运行时会话
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_image(self, image):
        # YOLOv5的预处理步骤通常包括：
        # 1. 将图像调整到模型期望的尺寸
        # 2. 将PIL图像转换为NumPy数组
        # 3. 归一化像素值
        # 4. 增加一个批次维度
        # 注意：这里的尺寸和归一化参数需要根据模型进行调整
        image = image.resize((640, 640))
        image_array = np.array(image).astype(np.float32)
        image_array /= 255.0  # 归一化
        image_array = np.expand_dims(image_array, axis=0)  # 增加批次维度
        image_array = image_array.transpose((0, 3, 1, 2))
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
        # 使用 transpose 函数来重新排列轴的顺序
        # 这里的 (0, 3, 1, 2) 表示：保持第一个维度不变，将第二个维度（颜色通道）移动到第二位，然后是宽度和高度
        image_array_transposed = image_array.transpose((0, 3, 1, 2))
        return image_array_transposed

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

class OCRPipeline:
    def __init__(self,use_angle_cls=True, lang="ch",use_gpu=True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang,use_gpu=use_gpu)  # 这里设置为中文识    

    def preprocess_image(self,image):
        return image

    def postprocess_results(self,original_image,result):

        draw_image = self.draw_text_on_white_image(original_image,result)
        # 准备JSON格式的数据结构
        ocr_results = {
            "draw_image":self.convertBase64(draw_image),
            "detections": []
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
    
    def convertBase64(self,image):
        # 将NumPy数组转换为PIL图像
        if isinstance(image, np.ndarray):
        # 如果是NumPy数组，使用PIL的fromarray函数将其转换为PIL图像
            image = Image.fromarray(image)
        # 创建一个字节流对象
        buffered = BytesIO()
        # 将PIL图像保存到字节流中，格式为'JPEG'或'PNG'
        image.save(buffered, format="JPEG")
        # 获取字节流的bytes
        byte_data = buffered.getvalue()
        # 对字节数据进行Base64编码
        base64_encoded = base64.b64encode(byte_data)
        # 将编码后的bytes转换为字符串
        base64_string = base64_encoded.decode('utf-8')
        return base64_string
    
    def __call__(self, image):
        image_deal = self.preprocess_image(image)
        results = self.ocr.ocr(image_deal)
        detections= self.postprocess_results(image,results[0])
        return detections


    