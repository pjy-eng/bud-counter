import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="Bud Counter Pro", layout="wide")

if 'roi_coords' not in st.session_state:
    st.session_state['roi_coords'] = None

# ==========================================
# 1. 核心算法库 (保持不变，稳健性强)
# ==========================================
def get_contour_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    return {"area": area, "circularity": circularity}

def process_and_count(img_gray, roi_coords, params):
    try:
        # --- A. 预处理 ---
        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # --- B. 图像分割 ---
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        k_size = int(params['open_kernel'])
        if k_size % 2 == 0: k_size += 1 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # --- C. 分水岭 ---
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        if dist_transform.max() == 0:
            return None, None, "图像预处理失败（全黑），请调整参数。"

        _, sure_fg = cv2.threshold(dist_transform, params['dist_ratio'] * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        # --- D. 提取模板特征 ---
        # Canvas 返回的坐标是 int，直接使用
        rx, ry, rw, rh = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
        
        # 提取模板 ROI
        roi_region_bin = opening[ry:ry+rh, rx:rx+rw]
        roi_cnts, _ = cv2.findContours(roi_region_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not roi_cnts:
             # 备用方案：原图 Otsu
            roi_raw = img_gray[ry:ry+rh, rx:rx+rw]
            _, roi_backup_thresh = cv2.threshold(roi_raw, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            roi_cnts, _ = cv2.findContours(roi_backup_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not roi_cnts:
            return None, None, "框选区域内没有检测到明显的 Bud，请重画。"

        template_cnt = max(roi_cnts, key=cv2.contourArea)
        tmpl_feats = get_contour_features(template_cnt)

        # --- E. 匹配筛选 ---
        final_buds = []
        unique_markers = np.unique(markers)
        for label in unique_markers:
            if label <= 1: continue 
            mask = np.zeros(img_gray.shape, dtype=np.uint8)
            mask[markers == label] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            c = max(cnts, key=cv2.contourArea)
            feats = get_contour_features(c)
            
            # 筛选逻辑
            area_ratio = abs(feats['area'] - tmpl_feats['area']) / (tmpl_feats['area'] + 1e-5)
            if area_ratio > params['area_tol']: continue 
            if feats['circularity'] < params['circ_thresh']: continue 
            final_buds.append(c)

        # --- F. 绘图 ---
        res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(res_img, final_buds, -1, (0, 0, 255), 2)
        # 画出质心
        for c in final_buds:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cv2.circle(res_img, (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), 3, (0, 255, 255), -1)
        
        return final_buds, res_img, tmpl_feats

    except Exception as e:
        return None, None, f"算法错误: {str(e)}"

# ==========================================
# 2. UI 布局
# ==========================================
st.sidebar.header("🎛️ 算法微调")
params = {
    'clahe_clip': st.sidebar.slider("对比度增强", 1.0, 5.0, 2.0),
    'open_kernel': st.sidebar.slider("去噪强度", 1, 9, 3),
    'dist_ratio': st.sidebar.slider("粘连分离", 0.1, 0.9, 0.5),
    'area_tol': st.sidebar.slider("面积容差", 0.1, 1.5, 0.5),
    'circ_thresh': st.sidebar.slider("圆度限制", 0.1, 1.0, 0.6)
}

st.title("🔬 Pro 级细胞 Bud 计数系统")

uploaded_file = st.file_uploader("1. 上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("2. 框选 ROI 模板")
        st.caption("请在下方图片中，用鼠标画一个矩形框住标准的 Bud。")
        
        # 使用 Canvas 替代 Plotly 进行画图，这是解决 TypeError 的唯一稳定方案
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_color="#00FF00",
            background_image=pil_img,
            update_streamlit=True,
            height=500, # 固定高度方便操作
            drawing_mode="rect",
            key="canvas",
        )

    with col2:
        st.subheader("3. 实时结果")
        
        # 检查 Canvas 是否有画图数据
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            # 获取最后一个画的框
            roi_obj = canvas_result.json_data["objects"][-1]
            roi_coords = {
                'left': int(roi_obj['left']), 'top': int(roi_obj['top']),
                'width': int(roi_obj['width']), 'height': int(roi_obj['height'])
            }
            
            # 实时计算
            with st.spinner("计算匹配中..."):
                buds, res_img, info = process_and_count(img_gray, roi_coords, params)

            if buds is not None:
                st.metric("计数结果", f"{len(buds)} 个")
                
                # 使用 Plotly 展示结果（支持放大查看）
                fig_res = px.imshow(res_img)
                fig_res.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=400)
                st.plotly_chart(fig_res, use_container_width=True)
                
                st.success(f"模板特征: 面积 {int(info['area'])} | 圆度 {info['circularity']:.2f}")
            else:
                st.error(info)
        else:
            st.info("👈 请在左侧画框以开始。")

else:
    st.info("👋 请上传图片。")
