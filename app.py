import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ------------------
# 一些工具函数
# ------------------

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def process_image(pil_image, roi):
    """
    pil_image: PIL Image (RGB)
    roi: dict with keys x, y, w, h  (基于原图像素坐标)
    """
    # PIL -> OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 预处理：CLAHE 提升对比 + 高斯滤波
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    img_blur = cv2.GaussianBlur(img_enhanced, (5, 5), 0)

    rx, ry, rw, rh = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    if rw <= 0 or rh <= 0:
        return img_bgr, 0, "ROI 尺寸无效"

    # 边界保护
    H, W = img_blur.shape
    rx = max(0, min(rx, W - 1))
    ry = max(0, min(ry, H - 1))
    rw = max(1, min(rw, W - rx))
    rh = max(1, min(rh, H - ry))

    roi_region = img_blur[ry : ry + rh, rx : rx + rw]

    # 在 ROI 内部进行 Otsu 阈值分割
    _, roi_thresh = cv2.threshold(
        roi_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img_bgr, 0, "ROI 中未检测到明显轮廓"

    # 取最大轮廓作为“标准芽”
    template_cnt = max(contours, key=cv2.contourArea)

    tmpl_area = cv2.contourArea(template_cnt)
    tmpl_perimeter = cv2.arcLength(template_cnt, True)
    tmpl_circularity = calculate_circularity(tmpl_area, tmpl_perimeter)
    tmpl_mean_intensity = cv2.mean(roi_region, mask=roi_thresh)[0]

    # 全图自适应阈值
    thresh_global = cv2.adaptiveThreshold(
        img_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        2,
    )

    # 形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_global = cv2.morphologyEx(
        thresh_global, cv2.MORPH_OPEN, kernel, iterations=2
    )

    all_contours, _ = cv2.findContours(
        thresh_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    matched_buds = []

    # 容差（可以后面做成侧边栏 slider）
    area_tol = 0.45  # 面积 ±45%
    circ_tol = 0.30  # 圆度 阈值系数
    intensity_tol = 0.35  # 灰度 ±35%

    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area < 3:  # 太小的直接跳过
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = calculate_circularity(area, perimeter)

        # 面积筛选
        if not (tmpl_area * (1 - area_tol) < area < tmpl_area * (1 + area_tol)):
            continue

        # 圆度筛选（只要求不比模板“差太多”）
        if circularity < tmpl_circularity * (1 - circ_tol):
            continue

        # 灰度筛选：计算该轮廓的平均灰度
        mask = np.zeros_like(img_gray)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
        mean_intensity = cv2.mean(img_gray, mask=mask)[0]
        if not (
            tmpl_mean_intensity * (1 - intensity_tol)
            < mean_intensity
            < tmpl_mean_intensity * (1 + intensity_tol)
        ):
            continue

        matched_buds.append(cnt)

    # 结果图绘制
    result_img = img_bgr.copy()
    cv2.drawContours(result_img, matched_buds, -1, (0, 0, 255), 2)
    cv2.rectangle(result_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    count = len(matched_buds)
    cv2.putText(
        result_img,
        f"Count: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3,
    )

    msg = f"模板面积={tmpl_area:.1f}, 圆度={tmpl_circularity:.3f}, 灰度={tmpl_mean_intensity:.1f}"
    return result_img, count, msg


# ------------------
# Streamlit 页面
# ------------------

st.set_page_config(page_title="Bud Counter (Streamlit + OpenCV)", layout="wide")
st.title("🔬 细胞芽（Bud）交互式计数 · Streamlit 版")
st.write(
    "步骤：上传显微图像 → 在图像上画一个矩形框住一个“标准芽” → 自动在全图中寻找相似芽并计数。"
)

# 侧边栏参数（你以后可以调）
with st.sidebar:
    st.header("参数 / 提示")
    st.markdown(
        "- 尽量框选一个**代表性、形态清晰**的芽\n"
        "- 不要框太大，避免包含太多背景\n"
        "- 如果检测结果太少/太多，可以之后把阈值做成 slider 来调"
    )

uploaded_file = st.file_uploader("📁 上传显微图像", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is None:
    st.info("请先上传一张显微图像。")
    st.stop()

# 读取并显示原图
pil_image = Image.open(uploaded_file).convert("RGB")
w, h = pil_image.size

st.subheader("① 原始图像 & ROI 框选")
st.write("在下方画布上选择 `矩形` 工具，框选一个芽作为模板。")

# 使用 drawable canvas 作为交互
canvas_result = st_canvas(
    fill_color="rgba(0, 255, 0, 0.2)",  # 填充色
    stroke_width=2,
    stroke_color="#00FF00",
    background_color="#000000",
    background_image=pil_image,
    update_streamlit=True,
    height=h,
    width=w,
    drawing_mode="rect",
    key="canvas",
)

roi = None
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    if objects:
        # 取最后一个矩形作为 ROI
        for obj in objects[::-1]:
            if obj.get("type") == "rect":
                left = obj.get("left", 0)
                top = obj.get("top", 0)
                width_rect = obj.get("width", 0)
                height_rect = obj.get("height", 0)

                # drawable-canvas 的坐标就是基于当前画布像素，所以和图像一一对应
                roi = {
                    "x": left,
                    "y": top,
                    "w": width_rect,
                    "h": height_rect,
                }
                break

col1, col2 = st.columns(2)

with col1:
    if roi:
        st.success(f"已选择 ROI: x={roi['x']:.1f}, y={roi['y']:.1f}, w={roi['w']:.1f}, h={roi['h']:.1f}")
    else:
        st.warning("请在上方画布中画一个矩形框选一个芽。")

with col2:
    run_btn = st.button("🚀 开始识别并计数")

st.subheader("② 识别结果")

if run_btn:
    if roi is None:
        st.error("未检测到 ROI，请先在画布上画一个矩形框选一个芽。")
    else:
        with st.spinner("正在用 OpenCV 进行形态学分析和匹配..."):
            result_bgr, count, debug_msg = process_image(pil_image, roi)

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        st.image(
            [pil_image, result_rgb],
            caption=["原图", f"识别结果（Count = {count}）"],
            use_column_width=True,
        )

        st.success(f"检测到 {count} 个相似候选芽。")
        st.caption("Debug 信息：" + debug_msg)
else:
    st.info("在选择 ROI 后，点击 “开始识别并计数” 按钮。")
