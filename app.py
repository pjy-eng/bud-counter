import streamlit as st
import numpy as np
import cv2
from PIL import Image

# =========================
# 小工具函数
# =========================

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def calc_circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0, area, peri
    circ = 4.0 * np.pi * area / (peri ** 2)
    return circ, area, peri

# =========================
# 模板驱动的候选筛选
# =========================

def template_guided_detection(img_gray, roi):
    """
    img_gray: HxW 灰度图
    roi: dict(x, y, w, h) 像素坐标
    返回: result_bgr, matched_cnts, template_info(str)
    """

    H, W = img_gray.shape
    x = max(0, min(roi["x"], W - 1))
    y = max(0, min(roi["y"], H - 1))
    w = max(5, min(roi["w"], W - x))
    h = max(5, min(roi["h"], H - y))

    roi_patch = img_gray[y:y + h, x:x + w]

    # ---- 1. 模板特征提取 ----
    # CLAHE + Otsu 只在 ROI 内做一次局部分割
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    roi_enh = clahe.apply(roi_patch)
    _, roi_thr = cv2.threshold(roi_enh, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(roi_thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, [], "ROI 内没有检测到明显轮廓，请稍微缩小或移动 ROI。"

    tmpl_cnt = max(cnts, key=cv2.contourArea)
    tmpl_circ, tmpl_area, tmpl_peri = calc_circularity(tmpl_cnt)
    mask = np.zeros_like(roi_patch, dtype=np.uint8)
    cv2.drawContours(mask, [tmpl_cnt], -1, 255, -1)
    tmpl_mean = cv2.mean(roi_patch, mask=mask)[0]

    template_info = f"模板: area={tmpl_area:.1f}, circ={tmpl_circ:.3f}, mean={tmpl_mean:.1f}"

    # ---- 2. 全图候选生成 ----
    clahe_full = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe_full.apply(img_gray)
    blur = cv2.GaussianBlur(enh, (5, 5), 0)

    # 自适应阈值 + 形态学开运算
    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)

    all_cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # ---- 3. 模板特征筛选 ----
    matched = []

    area_tol = 0.6      # 面积 ±60%
    circ_factor = 0.5   # 圆度至少为模板的 50%
    gray_tol = 0.35     # 灰度 ±35%

    for c in all_cnts:
        circ, area, peri = calc_circularity(c)
        if area < 10:
            continue

        # 面积相似度
        if not (tmpl_area * (1 - area_tol) < area < tmpl_area * (1 + area_tol)):
            continue

        # 圆度
        if circ < tmpl_circ * circ_factor:
            continue

        # 灰度相似度
        mask_c = np.zeros_like(img_gray, dtype=np.uint8)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        mean_c = cv2.mean(img_gray, mask=mask_c)[0]
        if not (tmpl_mean * (1 - gray_tol) < mean_c < tmpl_mean * (1 + gray_tol)):
            continue

        matched.append(c)

    # ---- 4. 可视化 ----
    result = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # ROI 矩形
    cv2.rectangle(result,
                  (x, y),
                  (x + w, y + h),
                  (0, 255, 0), 2)

    # 匹配到的 bud
    for c in matched:
        cv2.drawContours(result, [c], -1, (0, 0, 255), 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 3, (0, 255, 255), -1)

    cv2.putText(result, f"Count: {len(matched)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    return result, matched, template_info

# =========================
# Streamlit 界面
# =========================

st.set_page_config(page_title="Bud 计数（稳定版）", layout="wide")
st.title("🔬 Bud 模板驱动计数 · 稳定版（滑条选择 ROI）")

st.markdown(
    """
**使用步骤**  
1. 上传一张显微图像（tif / png / jpg）  
2. 调整滑条，选择一个包含单个 Bud 的矩形区域作为模板  
3. 点击“开始识别并计数”，查看红色轮廓 + 黄点 + 总数  
    """,
)

uploaded = st.file_uploader("📁 上传显微图像", type=["png", "jpg", "jpeg", "tif", "tiff"])

if not uploaded:
    st.info("请先上传一张图像。")
    st.stop()

# 读图
pil_img = Image.open(uploaded).convert("RGB")
img_bgr = pil_to_bgr(pil_img)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
H, W = img_gray.shape

# =========================
# ROI 选择（滑条）
# =========================

st.subheader("① 选择模板 ROI（滑条控制）")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.write("当前图像尺寸：", W, "×", H)

    # 以比例滑条避免超过边界
    x_ratio = st.slider("ROI 左上角 X（相对宽度）", 0.0, 0.9, 0.3, 0.01)
    y_ratio = st.slider("ROI 左上角 Y（相对高度）", 0.0, 0.9, 0.2, 0.01)
    w_ratio = st.slider("ROI 宽度（相对宽度）", 0.05, 0.9, 0.25, 0.01)
    h_ratio = st.slider("ROI 高度（相对高度）", 0.05, 0.9, 0.25, 0.01)

    x = int(x_ratio * W)
    y = int(y_ratio * H)
    w = int(w_ratio * W)
    h = int(h_ratio * H)

    roi = {"x": x, "y": y, "w": w, "h": h}

    preview = img_bgr.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(bgr_to_rgb(preview), caption="带 ROI 预览（绿色框）", use_container_width=True)

with col_right:
    st.write("ROI 像素坐标：")
    st.code(f"x={x}, y={y}, w={w}, h={h}", language="text")
    roi_valid = (w > 10 and h > 10)
    if not roi_valid:
        st.warning("ROI 太小，可能无法正确提取模板。")

st.subheader("② 识别结果")

run = st.button("🚀 开始识别并计数")

if run:
    if not roi_valid:
        st.error("ROI 尺寸太小，请增大宽度/高度。")
    else:
        with st.spinner("正在进行模板分析 + 全图匹配..."):
            result_bgr, matched_cnts, tmpl_info = template_guided_detection(
                img_gray, roi
            )
        if result_bgr is None:
            st.error(tmpl_info)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(bgr_to_rgb(img_bgr), caption="原图", use_container_width=True)
            with col2:
                st.image(bgr_to_rgb(result_bgr),
                         caption=f"识别结果（Count = {len(matched_cnts)}）",
                         use_container_width=True)
            st.success(f"模板特征：{tmpl_info}")
else:
    st.info("请调整 ROI 后点击上面的按钮进行识别。")
