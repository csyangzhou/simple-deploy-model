import argparse
import base64

import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify

from utils import image_utils
import global_var

app = Flask(__name__)


def parse_arg():
    parser = argparse.ArgumentParser(description='deploy model')
    parser.add_argument('--ip', type=str, default='192.168.1.6', help='host ip')
    parser.add_argument('--port', type=str, default=6666, help='host port')
    parser.add_argument('--is_use_gpu', type=bool, default=False, help='use gpu or cpu')
    parser.add_argument('--model_path', type=str, default='checkpoint/efficientdet(2).pt',
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


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # img base64
    # base64_str = image_utils.img_base64('images/shape_test/36002.jpg')
    # new_image = image_utils.process_image(base64_str)

    img = cv2.imread('images/shape_test/36003.jpg')
    height, width, _ = img.shape
    scale = 0
    if height > width:
        scale = height / 512
    else:
        scale = width / 512
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = cv2.resize(img, (512, 512))
    new_image = np.zeros((512, 512, 3))
    new_image[0:512, 0:512] = img
    new_image = torch.from_numpy(new_image)
    new_image = new_image.permute(2, 0, 1).float().unsqueeze(dim=0)

    with torch.no_grad():
        boxes = global_var.model(new_image)
    print(boxes.shape[0])
    if boxes.shape[0] > 0:
        print(boxes.shape[0])
    return data

def main():
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    args = parse_arg()
    global_var.model = load_model(args.model_path, args.is_use_gpu)
    app.run(host=args.ip, port=args.port)


if __name__ == "__main__":
    main()
