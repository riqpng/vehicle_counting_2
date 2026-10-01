from __future__ import division

from utils.utils import *
from utils.datasets import *
import cv2
import torch


def yolo_prediction(model, device, image, class_names):

    # Dimensões originais
    original_h, original_w = image.shape[:2]

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensiona diretamente com OpenCV
    image = cv2.resize(
        image,
        (416, 416),
        interpolation=cv2.INTER_LINEAR
    )

    # RGB -> Tensor
    imgs = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # Batch + device
    imgs = imgs.unsqueeze(0).to(device)

    # Inferência
    with torch.inference_mode():
        outputs = model(imgs)
        outputs = non_max_suppression(
            outputs,
            conf_thres=0.5,
            nms_thres=0.45
        )

    objects = []

    try:
        outputs = outputs[0].cpu()

        scale_x = original_w / 416
        scale_y = original_h / 416

        for output in outputs:

            item = []

            item.append(class_names[int(output[-1])])
            item.append(float(output[4]))

            x1, y1, x2, y2 = output[:4]

            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            x = int((x2 + x1) / 2)
            y = int((y1 + y2) / 2)

            box_w = x2 - x1
            box_h = y2 - y1

            item.append([x, y, box_w, box_h])

            objects.append(item)

    except Exception:
        pass

    return objects