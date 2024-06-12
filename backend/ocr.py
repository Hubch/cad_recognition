import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from utils import utils
import base64
from PIL import Image
from io import BytesIO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class OCRPipeline:
    def __init__(self,use_angle_cls=True, lang="ch",use_gpu=True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang,use_gpu=use_gpu)  # 这里设置为中文识    

    def converter(self,base64_str:str):
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)
        return img_array
    
    def preprocess_image(self,image):
        if isinstance(image,str):
          return self.converter(image)
        return image

    def postprocess_results(self,original_image,result):

        #draw_image = self.draw_text_on_white_image(original_image,result)
        # 准备JSON格式的数据结构
        ocr_results = {
            "draw_image":"utils.convertBase64(draw_image)",
            "detections": [],
            "texts":[line[1][0] for line in result]
        }

        # 遍历PaddleOCR的输出结果
        for line in result:
            # line格式为[[[x_min, y_min, x_max, y_max],(text, confidence)]]
            bbox = [[line[0]][0][0],[line[0]][0][1],[line[0]][0][2],[line[0]][0][3]]
            ocr_results["detections"].append({
                "bbox": tranXyxy(bbox), # 坐标
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
        detections= self.postprocess_results(image_deal,results[0])
        return detections

def tranXyxy(bbox):
    # 转换成 xyxy 格式
    x_min = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
    y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
    x_max = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
    y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
    return [x_min, y_min, x_max, y_max]    
