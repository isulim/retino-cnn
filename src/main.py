from litestar import Litestar, get, post

import torch
import onnxruntime
from PIL import Image

from torchvision import transforms


@get("/")
async def hello() -> str:
    return "Hello, World!"


@post("/classify")
async def classify() -> dict:

    ort = onnxruntime.InferenceSession("models/resnet34-model.onnx")
    img0 = Image.open("data/raw/0/0a38b552372d.jpg")
    img1 = Image.open("data/raw/1/0a9ec1e99ce4.jpg")

    try:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    except Exception as e:
        print(e)

    try:
        pt_0 = trans(img0).unsqueeze(0)
        pt_1 = trans(img1).unsqueeze(0)
    except Exception as e:
        print(e)

    try:
        out0 = ort.run(None, {"input.1": pt_0.numpy()})
        out1 = ort.run(None, {"input.1": pt_1.numpy()})
    except Exception as e:
        print(e)

    return {
        "res0": float(out0[0]),
        "res1": float(out1[0]),
    }

app = Litestar(
    route_handlers=[hello, classify]
)
