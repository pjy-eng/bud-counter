import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

# =========================
# 页面配置
# =========================
st.set_page_config(layout="wide")
st.title("🔬 Bud 自动识别系统（ResNet18 · 96×96 · 云端版）")

# =========================
# 模型下载配置（★你只需要改这里的 URL）
# =========================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1zavGWUgpoi5j3dSwNt4Xb-VQeD9WciAA"
MODEL_PATH = "ResNet18_window96_v2.pth"


# =========================
# 加载模型（自动下载）
# =========================
import requests

@st.cache_resource
def load_model():
    # ✅ 无论有没有旧文件，一律重新下载（防止HTML假文件污染）
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    with st.spinner("📥 正在从 Google Drive 强制重新下载模型权重..."):
        session = requests.Session()
        response = session.get(MODEL_URL, stream=True)

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                params = {"confirm": value}
                response = session.get(MODEL_URL, params=params, stream=True)
                break

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    # ✅ 下载完成后立即校验文件大小（防止再次下载到HTML）
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if file_size_mb < 10:
        raise RuntimeError(f"模型下载失败，当前文件只有 {file_size_mb:.2f} MB，极可能是 HTML 文件")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model, device


model, device = load_model()

# =========================
# 96×96 预处理（与你训练完全一致）
# =========================
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# =========================
# 滑窗候选生成
# =========================
def generate_patches(img_gray, step=48, win=96):
    patches = []
    coords = []
    h, w = img_gray.shape

    for y in range(0, h - win, step):
        for x in range(0, w - win, step):
            crop = img_gray[y:y+win, x:x+win]
            patches.append(crop)
            coords.append((x, y))

    return patches, coords

# =========================
# ResNet 分类
# =========================
@torch.no_grad()
def classify_patches(patches):
    probs = []

    for p in patches:
        pil = Image.fromarray(p).convert("L").convert("RGB")
        t = transform(pil).unsqueeze(0).to(device)
        out = model(t)
        prob = torch.softmax(out, 1)[0, 1].item()  # Bud 概率
        probs.append(prob)

    return probs

# =========================
# NMS 合并重复框
# =========================
def nms(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

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

# =========================
# Streamlit 主界面
# =========================
uploaded = st.file_uploader("📂 上传 TEM 图像", type=["png", "jpg", "tif"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns(2)
    col1.image(img_np, caption="原始图像", use_container_width=True)

    if st.button("🚀 开始自动识别 Bud"):
        with st.spinner("正在进行滑窗检测 + ResNet 推理..."):

            patches, coords = generate_patches(img_gray)
            probs = classify_patches(patches)

            boxes = []
            scores = []

            for (x, y), p in zip(coords, probs):
                if p > 0.85:    # ★你可以后续微调这个阈值
                    boxes.append((x, y, 96, 96))
                    scores.append(p)

            keep = nms(boxes, scores, threshold=0.25)

            result_img = img_np.copy()
            for i in keep:
                x, y, w, h = boxes[i]
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            col2.image(result_img,
                       caption=f"识别结果（Count = {len(keep)}）",
                       use_container_width=True)

            st.success(f"✅ 当前识别到 Bud 数量：{len(keep)}")

else:
    st.info("请先上传一张 TEM 图像。")
