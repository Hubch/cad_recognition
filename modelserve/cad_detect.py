from ray import serve
from ray.serve.handle import DeploymentHandle
from pipeline import YOLOv5ONNXPipeline
import base64   
import cv2
import numpy as np 
import torch
from PIL import Image
import paddleocr
from io import BytesIO
import json
import asyncio

onnx_model_path = 'models/yolov8n-seg.onnx'  # ONNX模型文件的路径

import logging

# 配置日志格式和级别
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@serve.deployment
def imageconverts(image: str):
    # 将base64格式的数据解码为二进制格式
    binary_data = base64.b64decode(image)
    # 将二进制数据转换为OpenCV格式的图片对象
    img = cv2.imdecode(np.frombuffer(binary_data, np.uint8), cv2.IMREAD_COLOR)
    # 调整图片大小
    img_size = (640, 640)  # 设置输出图片大小
    img_resized = cv2.resize(img, img_size)
    # 将图片转换为PyTorch张量
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)
    return img_tensor

@serve.deployment
class ObjectDetect:
    def __init__(self,converter:DeploymentHandle):
        self.converter = converter
        self.model = YOLOv5ONNXPipeline(onnx_model_path)
        self.logger = logging.getLogger(__name__)
        
    async def detect(self,base64_str:str):
        image = await self.converter.remote(base64_str)
        self.logger.info(f"image type：{type(image)}")
        results = self.model(image)
        return results
    
    async def __call__(self, base64_str: str):
        return await self.detect(base64_str)
    

@serve.deployment
def converterRGB(base64_str:str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

@serve.deployment
def converter(base64_str:str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    img_array = np.array(image)
    return img_array

@serve.deployment
class OCRTransform:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        # 使用PaddleOCR进行OCR识别
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")  # 这里设置为中文识    

    async def transform(self, base64_str: str) -> str:
        image = await self.downloader.remote(base64_str)
        results = self.ocr.ocr(image)
        return results

    async def __call__(self, base64_str: str):
        return await self.transform(base64_str)

@serve.deployment
class CADDetect:
    def __init__(
        self, detect_responder: DeploymentHandle, ocr_responder: DeploymentHandle
    ):
        self.detect_responder = detect_responder
        self.ocr_responder = ocr_responder
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self,http_request):
        try:
            request = await http_request.json()
            image = request["image"]
            detect_coro = self.detect_responder.remote(image)
            ocr_coro = self.ocr_responder.remote(image)
            detect_result, ocr_result = await asyncio.gather(detect_coro, ocr_coro)
            response = {"org_image":image,"detect_result":detect_result,"ocr_result":ocr_result}
            json_response = json.dumps(response)
            self.logger.info(f"cad detect result: {json_response}")
            return await json_response
        except Exception as e:
            self.logger.error(f"Error : {e}")
            return json.dumps({"error":"服务异常"})
            # 根据需要处理异常
        
    

objectdetect_responder = ObjectDetect.bind(converterRGB.bind())
ocrtransform_responder = OCRTransform.bind(converter.bind())
cad_detect = CADDetect.options(route_prefix="/caddetect").bind(objectdetect_responder, ocrtransform_responder)