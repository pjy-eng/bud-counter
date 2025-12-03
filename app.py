import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# =========================
# 工具函数
# =========================

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def process_image(pil_image, roi):
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    img_blur = cv2.GaussianBlur(img_enhanced, (5, 5), 0)

    rx, ry, rw, rh = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])

    if rw <= 5 or rh <= 5:
        return img_bgr, 0, "ROI 太小"

    H, W = img_blur.shape
    rx = max(0, min(rx, W - 1))
    ry = max(0, min(ry, H - 1))
    rw = max(1, min(rw, W - rx))
    rh = max(1, min(rh, H - ry))

    roi_region = img_blur[ry:ry + rh, rx:rx + rw]

    _, roi_thresh = cv2.threshold(
        roi_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img_bgr, 0, "ROI 中未检测到目标"

    template_cnt = max(contours, key=cv2.contourArea)

    tmpl_area = cv2.contourArea(template_cnt)
    tmpl_perimeter = cv2.arcLength(template_cnt, True)
    tmpl_circ = calculate_circularity(tmpl_area, tmpl_perimeter)
    tmpl_mean = cv2.mean(roi_region, mask=roi_thresh)[0]

    thresh_global = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_global = cv2.morphologyEx(
        thresh_global, cv2.MORPH_OPEN, kernel, iterations=2
    )

    all_contours, _ = cv2.findContours(
        thresh_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    matched = []

    area_tol = 0.45
    circ_tol = 0.30
    gray_tol = 0.35

    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area < 8:
            continue

        per = cv2.arcLength(cnt, True)
        circ = calculate_circularity(area, per)

        if not (tmpl_area * (1 - area_tol) < area < tmpl_area * (1 + area_tol)):
            continue

        if circ < tmpl_circ * (1 - circ_tol):
            continue

        mask = np.zeros_like(img_gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(img_gray, mask=mask)[0]

        if not (tmpl_mean * (1 - gray_tol) < mean_val < tmpl_mean * (1 + gray_tol)):
            continue

        matched.append(cnt)

    result_img = img_bgr.copy()
    cv2.drawContours(result_img, matched, -1, (0, 0, 255), 2)
    cv2.rectangle(result_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    count = len(matched)
    cv2.putText(
        result_img, f"Count: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3
    )

    msg = f"模板面积={tmpl_area:.1f}, 圆度={tmpl_circ:.3f}, 灰度={tmpl_mean:.1f}"
    return result_img, count, msg


# =========================
# Streamlit 页面
# =========================

st.set_page_config(page_title="Bud Counter", layout="wide")
st.title("🔬 细胞芽（Bud）在线计数 · 兼容版")

uploaded_file = st.file_uploader(
    "📁 上传显微图像", type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is None:
    st.info("请先上传一张显微图像。")
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")
img_w, img_h = pil_image.size

st.subheader("① 原始图像（仅用于参考）")
st.image(pil_image, use_column_width=True)

st.subheader("② ROI 框选（与原图完全等比例映射）")
st.write("请在下方白色画布中 **按照原图位置比例** 框选一个芽")

display_w = 600
scale = display_w / img_w
display_h = int(img_h * scale)

canvas_result = st_canvas(
    fill_color="rgba(0, 255, 0, 0.2)",
    stroke_width=2,
    stroke_color="#00FF00",
    background_color="white",
    update_streamlit=True,
    height=display_h,
    width=display_w,
    drawing_mode="rect",
    key="canvas",
)

roi = None
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    if objects:
        for obj in objects[::-1]:
            if obj.get("type") == "rect":
                roi = {
                    "x": int(obj.get("left") / scale),
                    "y": int(obj.get("top") / scale),
                    "w": int(obj.get("width") / scale),
                    "h": int(obj.get("height") / scale),
                }
                break

if roi:
    st.success(f"映射 ROI(px)：x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}")
else:
    st.warning("请在画布中框选一个芽作为模板")

if st.button("🚀 开始识别并计数"):
    if roi is None:
        st.error("未检测到 ROI")
    else:
        with st.spinner("正在计算..."):
            result_bgr, count, debug = process_image(pil_image, roi)

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        st.subheader("③ 识别结果")
        st.image(
            [pil_image, result_rgb],
            caption=["原图", f"检测结果（Count={count}）"],
            use_column_width=True,
        )

        st.success(f"✅ 检测到 {count} 个芽")
        st.caption("Debug：" + debug)
