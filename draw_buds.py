import cv2
import numpy as np
from PIL import Image
import io


def draw_buds(img_bytes, buds):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)

    for b in buds:
        x, y, r = int(b["x"]), int(b["y"]), int(b["r"])
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    return img
