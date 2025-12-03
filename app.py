import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 图像处理核心（分水岭）
# =========================

def watershed_bud_segmentation(img_gray):
    # 去噪
    denoise = cv2.fastNlMeansDenoising(img_gray, None, h=10)

    # 对比度增强
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enh = clahe.apply(denoise)

    # Otsu 二值化
    _, bw = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)

    # 距离变换
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 前景种子
    _, sure_fg = cv2.threshold(dist_norm, 0.4, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)

    # 背景
    sure_bg = cv2.dilate(bw, kernel, iterations=3)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # 连通域标记
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # 提取每个 bud 的轮廓
    buds = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.uint8(markers == label)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 30:
                buds.append(c)

    return buds


# =========================
# 将 Plotly ROI 转换回像素
# =========================

def parse_plotly_roi(relayout_data, img_w, img_h):
    try:
        x0 = int(relayout_data["shapes[0].x0"])
        y0 = int(relayout_data["shapes[0].y0"])
        x1 = int(relayout_data["shapes[0].x1"])
        y1 = int(relayout_data["shapes[0].y1"])
        return {
            "x": min(x0, x1),
            "y": min(y0, y1),
            "w": abs(x1 - x0),
            "h": abs(y1 - y0),
        }
    except:
        return None


# =========================
# Streamlit 页面
# =========================

st.set_page_config(layout="wide")
st.title("🔬 Bud 在线计数（Plotly 交互 + 分水岭）")

uploaded_file = st.file_uploader("📁 上传显微图像", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is None:
    st.info("请先上传一张图像")
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")
img_w, img_h = pil_image.size
img_gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

# =========================
# Plotly 交互 ROI
# =========================

st.subheader("① 在原图上直接框选一个 Bud 作为模板")

fig = px.imshow(pil_image)
fig.update_layout(
    dragmode="drawrect",
    newshape=dict(line_color="lime"),
    margin=dict(l=0, r=0, t=0, b=0)
)

plotly_event = st.plotly_chart(fig, use_container_width=True)

roi = None
if plotly_event and hasattr(plotly_event, "relayout_data"):
    roi = parse_plotly_roi(plotly_event.relayout_data, img_w, img_h)

if roi:
    st.success(f"ROI: x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}")
else:
    st.warning("请在图像上直接画一个矩形")

# =========================
# 分水岭识别
# =========================

if st.button("🚀 开始识别并计数"):
    with st.spinner("正在进行分水岭分割与计数..."):

        buds = watershed_bud_segmentation(img_gray)

        result = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for c in buds:
            cv2.drawContours(result, [c], -1, (0, 0, 255), 2)

        # ROI 标注
        if roi:
            cv2.rectangle(
                result,
                (roi["x"], roi["y"]),
                (roi["x"] + roi["w"], roi["y"] + roi["h"]),
                (0, 255, 0),
                2,
            )

        count = len(buds)

        st.subheader("② 分水岭识别结果")
        st.image(
            [pil_image, result[:, :, ::-1]],
            caption=["原图", f"识别结果（Count={count}）"],
            use_column_width=True,
        )
        st.success(f"✅ 当前检测到 {count} 个 bud")
