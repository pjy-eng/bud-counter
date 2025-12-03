import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------
# Streamlit 页面配置
# -------------------------------
st.set_page_config(layout="wide")
st.title("🔬 Bud 自动识别系统（ResNet 模型版）")

# -------------------------------
# 1. 加载你的 ResNet 模型
# -------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# 2. 滑窗候选生成
# -------------------------------
def generate_patches(img_gray, step=64, win=128):
    patches = []
    coords = []
    h, w = img_gray.shape
    for y in range(0, h - win, step):
        for x in range(0, w - win, step):
            crop = img_gray[y:y+win, x:x+win]
            patches.append(crop)
            coords.append((x, y))
    return patches, coords

# -------------------------------
# 3. ResNet 判断 Bud
# -------------------------------
@torch.no_grad()
def classify_patches(patches):
    preds = []
    for p in patches:
        pil = Image.fromarray(p).convert("L").convert("RGB")
        t = transform(pil).unsqueeze(0).to(device)
        out = model(t)
        prob = torch.softmax(out, 1)[0, 1].item()  # Bud 概率
        preds.append(prob)
    return preds

# -------------------------------
# 4. NMS 合并重复检测
# -------------------------------
def nms(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    areas = boxes[:,2] * boxes[:,3]

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < threshold)[0]
        order = order[inds + 1]
    return keep

# -------------------------------
# 5. Streamlit 主界面
# -------------------------------
uploaded = st.file_uploader("上传 TEM 图像", type=["png", "jpg", "tif"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns(2)
    col1.image(img_np, caption="原图", use_container_width=True)

    if st.button("🚀 开始自动识别"):
        with st.spinner("正在进行滑窗 + ResNet 预测..."):
            patches, coords = generate_patches(img_gray)
            probs = classify_patches(patches)

            boxes = []
            scores = []
            for (x, y), p in zip(coords, probs):
                if p > 0.9:  # 置信度阈值
                    boxes.append((x, y, 128, 128))
                    scores.append(p)

            keep = nms(boxes, scores)

            result_img = img_np.copy()
            for i in keep:
                x, y, w, h = boxes[i]
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0,255,0), 2)

            col2.image(result_img, caption=f"识别结果 Count={len(keep)}", use_container_width=True)
            st.success(f"✅ 识别 Bud 数量：{len(keep)}")

