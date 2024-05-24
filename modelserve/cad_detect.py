from ray import serve
from ray.serve.handle import DeploymentHandle
from pipeline import YOLOv5ONNXPipeline,OCRPipeline
import base64   
import cv2
import numpy as np 
import torch
from PIL import Image
from io import BytesIO
import json
import asyncio
from utils.convert import CustomJSONEncoder

onnx_model_path = 'models/yolov5l_cad_gpu-1.0.onnx'  # ONNX模型文件的路径

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
def CvConverter(base64_str:str):
    # return BGR image which required be convert to RGB in post 
    binary_data = base64.b64decode(base64_str)
    return cv2.imdecode(np.frombuffer(binary_data, np.uint8), cv2.IMREAD_COLOR)

@serve.deployment
def converterRGB(base64_str:str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

@serve.deployment
def converter(base64_str:str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    img_array = np.array(image)
    return img_array

@serve.deployment(ray_actor_options={"num_cpus": 2, "num_gpus": 1},health_check_timeout_s=60,health_check_period_s=60)
class OCRTransform:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        # 使用PaddleOCR进行OCR识别
        self.ocr = OCRPipeline(use_angle_cls=True, lang="ch",use_gpu=False)  # 这里设置为中文识    

    async def transform(self, base64_str: str) -> str:
        image = await self.downloader.remote(base64_str)
        results = self.ocr(image)
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
            ocr_coro = self.ocr_responder.remote(image)
            detect_coro = self.detect_responder.remote(image)
            ocr_result,detect_result = await asyncio.gather(ocr_coro,detect_coro)
            response = {"org_image":image,"detect_result":detect_result,"ocr_result":ocr_result}
            json_response = json.dumps(response,cls = CustomJSONEncoder)
            self.logger.info(f"cad detect result: {json_response}")
            return  json_response
            # return response
        except Exception as e:
            self.logger.error(f"Error : {e}")
            return json.dumps({"error":"服务异常"})
            # 根据需要处理异常
        

objectdetect_responder = ObjectDetect.bind(CvConverter.bind())
ocrtransform_responder = OCRTransform.bind(converter.bind())
cad_detect = CADDetect.options(route_prefix="/caddetect").bind(objectdetect_responder, ocrtransform_responder)