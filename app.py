import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="高精度 Bud 计数系统", layout="wide")

# 初始化 Session State
if 'roi_coords' not in st.session_state:
    st.session_state['roi_coords'] = None
if 'processed_result' not in st.session_state:
    st.session_state['processed_result'] = None

# ==========================================
# 1. 核心算法库
# ==========================================

def get_contour_features(contour):
    """计算轮廓特征"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    return {"area": area, "circularity": circularity}

def process_and_count(img_gray, roi_coords, params):
    """
    核心处理流程：
    1. 预处理 (CLAHE + Blur)
    2. 动态阈值分割 (基于用户滑块)
    3. 分水岭分离粘连
    4. 模板特征匹配筛选
    """
    # --- A. 预处理 ---
    # CLAHE 增强对比度 (应对电镜图的关键)
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # --- B. 图像分割 (提取前景) ---
    # 使用二值化找到大概区域
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学开运算：去除小白点噪声
    kernel_size = int(params['open_kernel'])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- C. 分水岭算法 (分离粘连) ---
    # 确定的背景
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 确定的前景 (使用距离变换)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 这里的 0.5 是一个经验值，可以用滑块控制灵敏度
    _, sure_fg = cv2.threshold(dist_transform, params['dist_ratio'] * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # --- D. 提取模板特征 ---
    rx, ry, rw, rh = roi_coords['x'], roi_coords['y'], roi_coords['w'], roi_coords['h']
    # 简单的从 mask 中提取 ROI 区域对应的特征，这里简化为取 ROI 框内最大的轮廓
    # 为了更准，我们直接分析 ROI 区域的图像
    roi_region_bin = opening[ry:ry+rh, rx:rx+rw]
    roi_cnts, _ = cv2.findContours(roi_region_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not roi_cnts:
        return None, "模板区域内未检测到目标，请调整阈值或重画 ROI"
    
    template_cnt = max(roi_cnts, key=cv2.contourArea)
    tmpl_feats = get_contour_features(template_cnt)

    # --- E. 全图匹配筛选 ---
    final_buds = []
    candidates_count = 0
    
    # 遍历分水岭标记的所有区域
    for label in np.unique(markers):
        if label <= 1: continue # 跳过背景
        
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        
        c = max(cnts, key=cv2.contourArea)
        feats = get_contour_features(c)
        candidates_count += 1
        
        # 1. 面积筛选
        area_ratio = abs(feats['area'] - tmpl_feats['area']) / tmpl_feats['area']
        if area_ratio > params['area_tol']:
            continue # 面积差异太大
            
        # 2. 圆度筛选
        if feats['circularity'] < params['circ_thresh']:
            continue # 形状不够圆
            
        final_buds.append(c)

    # --- F. 绘图 ---
    res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # 画所有检测到的
    cv2.drawContours(res_img, final_buds, -1, (0, 0, 255), 2) # 红色轮廓
    
    # 画 ROI
    cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
    
    # 画质心
    for c in final_buds:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(res_img, (cX, cY), 3, (0, 255, 255), -1) # 黄点

    return final_buds, res_img, tmpl_feats

# ==========================================
# 2. 辅助函数
# ==========================================
def parse_plotly_relayout(relayout_data):
    if not relayout_data: return None
    if "shapes[0].x0" in relayout_data:
        x0, x1 = relayout_data["shapes[0].x0"], relayout_data["shapes[0].x1"]
        y0, y1 = relayout_data["shapes[0].y0"], relayout_data["shapes[0].y1"]
    elif "shapes" in relayout_data and len(relayout_data["shapes"]) > 0:
        last = relayout_data["shapes"][-1]
        x0, x1, y0, y1 = last["x0"], last["x1"], last["y0"], last["y1"]
    else: return None
    return {"x": int(min(x0, x1)), "y": int(min(y0, y1)), "w": int(abs(x1 - x0)), "h": int(abs(y1 - y0))}

def reset_callback():
    st.session_state['roi_coords'] = None
    st.session_state['processed_result'] = None

# ==========================================
# 3. 界面布局
# ==========================================
st.sidebar.header("🎛️ 算法微调 (关键)")
st.sidebar.info("💡 如果识别不准，请调整以下参数直到满意。")

# --- 关键参数滑块 ---
params = {
    'clahe_clip': st.sidebar.slider("对比度增强 (CLAHE)", 1.0, 5.0, 2.0, 0.5),
    'open_kernel': st.sidebar.slider("去噪强度 (Kernel)", 1, 7, 3, 2),
    'dist_ratio': st.sidebar.slider("粘连分离灵敏度", 0.1, 0.9, 0.5, 0.05, help="越小分得越细，越大越容易粘连"),
    'area_tol': st.sidebar.slider("面积容差 (±%)", 0.1, 1.0, 0.5, 0.05, help="允许目标大小与模板差异的程度"),
    'circ_thresh': st.sidebar.slider("最小圆度限制", 0.1, 1.0, 0.6, 0.05, help="越接近1越圆，排除长条形噪点")
}

st.title("🔬 高精度 Bud 计数系统 (Pro)")

uploaded_file = st.file_uploader("1. 上传图片", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("2. 框选 ROI 模板")
        fig = px.imshow(pil_img)
        fig.update_layout(dragmode='drawrect', newshape=dict(line_color='cyan', line_width=3), height=500, margin=dict(l=0, r=0, t=0, b=0))
        
        if st.session_state['roi_coords']:
            r = st.session_state['roi_coords']
            fig.add_shape(type="rect", x0=r['x'], y0=r['y'], x1=r['x']+r['w'], y1=r['y']+r['h'], line=dict(color="green", width=4))
        
        relayout_data = st.plotly_chart(fig, use_container_width=True, on_select="ignore")
        if relayout_data and ("shapes" in relayout_data or "shapes[0].x0" in relayout_data):
            new_roi = parse_plotly_relayout(relayout_data)
            if new_roi:
                st.session_state['roi_coords'] = new_roi
                st.rerun()

    with col2:
        st.subheader("3. 结果面板")
        if st.session_state['roi_coords']:
            st.button("🔄 重新画框", on_click=reset_callback)
            
            # 只要有 ROI，就自动开始尝试计算（配合滑块实时更新）
            # 或者你可以选择保留“开始计算”按钮，但我建议为了调参体验，直接实时计算
            buds, res_img, tmpl_feats = process_and_count(img_gray, st.session_state['roi_coords'], params)
            
            if buds is not None:
                st.metric("✅ 计数结果", f"{len(buds)} 个")
                st.image(res_img, caption="红色: 识别结果 | 黄点: 质心", use_column_width=True)
                st.caption(f"模板面积: {int(tmpl_feats['area'])} px | 模板圆度: {tmpl_feats['circularity']:.2f}")
            else:
                st.error(res_img) # 显示错误信息
        else:
            st.info("👈 请先在左侧图上框选一个标准 Bud")
