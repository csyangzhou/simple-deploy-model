import argparse

import torch
from flask import Flask, request, jsonify

import global_var

app = Flask(__name__)


def parse_arg():
    parser = argparse.ArgumentParser(description='deploy model')
    parser.add_argument('--ip', type=str, default='192.168.1.6', help='host ip')
    parser.add_argument('--port', type=str, default=6666, help='host port')
    parser.add_argument('--is_use_gpu', type=bool, default=False, help='use gpu or cpu')
    parser.add_argument('--model_path', type=str, default='checkpoint/DSTN_448.pt',
                        help='model path for deploying')
    return parser.parse_args()


def load_model(model_path, is_use_gpu):
    """
    :param model_path:
    :return:
    """
    model = torch.jit.load(model_path)
    model.eval()
    model.cuda() if is_use_gpu else model.cpu()
    return model
    # print(model)


@app.route("/predict", methods=['POST','GET'])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    return data

def main():
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    args = parse_arg()
    global_var.model = load_model(args.model_path, args.is_use_gpu)
    app.run(host=args.ip, port=args.port)


if __name__ == "__main__":
    main()
