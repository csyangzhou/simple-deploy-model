import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms

import main
import global_var


# 图片文件打开为base64
def img_base64(img_path):
    with open(img_path, "rb") as f:
        base64_str = base64.b64encode(f.read())
        return base64_str


# PIL转cv2
def pil_cv2(img_path):
    image = Image.open(img_path)
    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    return img


# base64转PIL
def base64_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image

"""
base64 和 cv2
"""
def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.frombuffer(imgString,np.uint8)
    image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return image

def process_image(img_base64_str):
    img = base64_cv2(img_base64_str)
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
    return new_image, scale


if __name__ == "__main__":
    global_var.model = main.load_model("../checkpoint/efficientdet.pt", False)
    img_base64_str = img_base64('../images/shape_test/36001.jpg')
    img = process_image(img_base64_str)
    with torch.no_grad():
        boxes = global_var.model(img)
    if boxes.shape[0] > 0:
        for box_id in range(boxes.shape[0]):
            xmin, ymin, xmax, ymax = boxes[box_id, :]

    print(boxes.size())



