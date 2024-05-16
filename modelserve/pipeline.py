import onnxruntime
from PIL import Image
import numpy as np
import logging
import torch

class YOLOv5ONNXPipeline:
    def __init__(self, onnx_model_path):
        # 初始化ONNX运行时会话
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.logger = logging.getLogger(__name__)

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
