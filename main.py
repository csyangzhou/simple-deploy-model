import argparse
import base64

import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from torchvision import transforms
import json

from utils import image_utils
import global_var

app = Flask(__name__)


def parse_arg():
    parser = argparse.ArgumentParser(description='deploy model')
    parser.add_argument('--ip', type=str, default='192.168.1.6', help='host ip')
    parser.add_argument('--port', type=str, default=6666, help='host port')
    parser.add_argument('--is_use_gpu', type=bool, default=False, help='use gpu or cpu')
    parser.add_argument('--model_path', type=str, default='checkpoint/efficientdet(4).jit',
                        help='model path for deploying')
    return parser.parse_args()


def load_model(model_path, is_use_gpu):
    """
    :param model_path:
    :return:
    """
    print(model_path)
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    model.cuda() if is_use_gpu else model.cpu()
    return model
    # print(model)

def parse_image_data():
    """
    获取appserver传过来的base64数据
    :return:
    """
    b64_data = None
    error_msg = None
    if not request.is_json:
        error_msg = "Error.Need json data!"
        return b64_data, error_msg
    req_body = request.data
    json_data = json.loads(req_body)
    b64_data = json_data["image"]
    return b64_data, error_msg

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"error": 1, "predict_res":[]}
    # img base64
    # base64_str = image_utils.img_base64('images/shape_test/36002.jpg')
    base64_str, error_msg = parse_image_data()
    new_image, scale = image_utils.process_image(base64_str)
    with torch.no_grad():
        scores, boxes = global_var.model(new_image)
    if boxes.shape[0] > 0:
        for box_id in range(boxes.shape[0]):
            pred_prob = float(scores[box_id])
            if pred_prob < global_var.cls_threshold:
                continue
            data["error"] = 0
            print(boxes[box_id, :].numpy())
            box = [int(boxes[box_id, :].numpy()[0] * scale) , int(boxes[box_id, :].numpy()[1] * scale), int(boxes[box_id, :].numpy()[2] * scale), int(boxes[box_id, :].numpy()[3] * scale)]
            data["predict_res"].append(box)
    print(str(data))
    return data


def main():
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    args = parse_arg()
    print(args)
    global_var.model = load_model(args.model_path, args.is_use_gpu)
    app.run(host=args.ip, port=args.port)


if __name__ == "__main__":
    main()
