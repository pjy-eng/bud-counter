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

# 初始化 session state
if 'roi_coords' not in st.session_state:
    st.session_state['roi_coords'] = None

# ==========================================
# 1. 核心算法库 (保持不变)
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

        # 这里的 clipLimit 由用户滑块控制
        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # --- B. 图像分割 ---
        # Otsu 二值化
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学去噪 (确保核大小为奇数)
        k_size = int(params['open_kernel'])
        if k_size % 2 == 0: k_size += 1 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # --- C. 分水岭 (分离粘连) ---
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # 防止全黑崩溃
        dist_max = dist_transform.max()
        if dist_max == 0:
            return None, None, "图像预处理失败（全黑），请降低去噪强度或调整对比度。"

        _, sure_fg = cv2.threshold(dist_transform, params['dist_ratio'] * dist_max, 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        # --- D. 提取模板特征 ---
        rx, ry, rw, rh = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
        
        # 提取 ROI
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
        # 画轮廓
        cv2.drawContours(res_img, final_buds, -1, (0, 0, 255), 2)
        # 画 ROI
        cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        # 画质心
        for c in final_buds:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cv2.circle(res_img, (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), 3, (0, 255, 255), -1)
        
        return final_buds, res_img, tmpl_feats

    except Exception as e:
        return None, None, f"算法内部错误: {str(e)}"

# ==========================================
# 2. UI 布局
# ==========================================
st.sidebar.header("🎛️ 算法微调")
st.sidebar.markdown("如果识别不准，请尝试调整以下参数：")

params = {
    'clahe_clip': st.sidebar.slider("对比度增强 (CLAHE)", 1.0, 5.0, 2.0, help="值越大，图像对比度越高"),
    'open_kernel': st.sidebar.slider("去噪强度", 1, 9, 3, help="值越大，噪点越少，但可能丢失小目标"),
    'dist_ratio': st.sidebar.slider("粘连分离灵敏度", 0.1, 0.9, 0.5, help="越小分得越细"),
    'area_tol': st.sidebar.slider("面积容差 (±%)", 0.1, 1.5, 0.5, help="允许目标大小与模板的差异程度"),
    'circ_thresh': st.sidebar.slider("最小圆度限制", 0.1, 1.0, 0.6, help="值越大，要求目标越圆")
}

st.title("🔬 Pro 级细胞 Bud 计数系统")

uploaded_file = st.file_uploader("1. 上传显微图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("2. 框选 ROI 模板")
        st.caption("请在下方图片中，用鼠标画一个 **矩形** 框住一个标准的 Bud。")
        
        # Streamlit Canvas 组件
        # 注意：这里的 height 和 width 最好根据图片比例动态调整，这里为了简单设为固定
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",  # 填充色 (半透明绿)
            stroke_color="#00FF00",             # 边框色 (绿)
            background_image=pil_img,           # 背景图
            update_streamlit=True,
            height=500,                         # 画布高度
            drawing_mode="rect",                # 只允许画矩形
            key="canvas",
        )

    with col2:
        st.subheader("3. 实时分析结果")
        
        # 检查是否画了框
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            # 获取最后一个画的对象
            obj = canvas_result.json_data["objects"][-1]
            roi_coords = {
                'left': int(obj['left']), 
                'top': int(obj['top']),
                'width': int(obj['width']), 
                'height': int(obj['height'])
            }
            
            # 只有当框的大小有效时才计算
            if roi_coords['width'] > 0 and roi_coords['height'] > 0:
                with st.spinner("正在匹配相似目标..."):
                    buds, res_img, info = process_and_count(img_gray, roi_coords, params)

                if buds is not None:
                    st.metric("✅ 计数结果", f"{len(buds)} 个")
                    
                    # 使用 Plotly 展示大图，方便缩放查看细节
                    fig = px.imshow(res_img)
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"模板数据: 面积 {int(info['area'])} | 圆度 {info['circularity']:.2f}")
                else:
                    st.error(f"⚠️ {info}")
        else:
            st.info("👈 等待框选...")

else:
    st.info("👋 请先上传一张图片。")
