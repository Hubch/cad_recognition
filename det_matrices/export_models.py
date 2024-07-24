import os 
import sys
import json
import argparse

import torch
import torch.onnx
import onnx

from _utils.models import get_student_model

def load_det_model(model_path):
    yolov5_path = os.path.join(os.getcwd(), 'yolov5')
    if yolov5_path not in sys.path:
        sys.path.append(yolov5_path)

    from yolov5.models.common import DetectMultiBackend
    model = DetectMultiBackend(model_path)
    return model

def load_class_mapping(class_map_path):
    with open(class_map_path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', type=str, required=True, help='yolov5 or match')
    parser.add_argument('--weight_path', type=str, help='full_path for yolov5, no-suffix for matching', default='./weights/cad_hybrid')
    parser.add_argument('--onnx_path', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model
    weight_path = args.weight_path
    onnx_path = args.onnx_path

    if model_type == 'yolov5':
        model = load_det_model(weight_path)
        model.eval()

        dummy_input = torch.randn((1, 3, 640, 640))
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=True,
            opset_version=12,
            input_names=["inputs"],
            output_names=["output"],
            dynamic_axes={"inputs": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 1: "anchors"}}
        )

        # model check
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)  # check onnx model
        
        onnx.save(model_onnx, onnx_path)
        print(f"YOLOv5 model has been converted to ONNX and saved at {onnx_path}")
    
    elif model_type == 'match':
        class_to_idx = load_class_mapping('./weights/class_map.json')
        model = get_student_model(num_classes=len(class_to_idx), dropout_p=0.3, init=False, update=False, weight_path=weight_path, cuda=False)
        model.eval()
    
        dummy_input = torch.randn((1, 1, 32, 32))
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=True,
            opset_version=12,
            input_names=["input"],
            output_names=["feature", "logits"],
            dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}, 
                          "feature": {0: "batch_size"}, 
                          "logits": {0: "batch_size"}}
        )

        # model check
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)  # check onnx model

        onnx.save(model_onnx, onnx_path)
        print(f"Custom model has been converted to ONNX and saved at {onnx_path}")

    else:
        print('--model must has the velue of \'yolov5\' or \'match\'. do nothing')
        sys.exit(1)
      