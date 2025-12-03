import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 0. 全局配置与状态初始化
# ==========================================
st.set_page_config(page_title="Template-Guided Bud Counter", layout="wide")

# 初始化 Session State 用于存储 ROI 和 处理状态
if 'roi_coords' not in st.session_state:
    st.session_state['roi_coords'] = None  # 格式: {'x':, 'y':, 'w':, 'h':}
if 'processed_result' not in st.session_state:
    st.session_state['processed_result'] = None

# ==========================================
# 1. 核心算法库 (特征提取 + 匹配)
# ==========================================

def get_contour_features(contour, img_gray_roi=None):
    """
    计算轮廓的几何特征：面积、周长、圆度、平均灰度(可选)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # 圆度计算: 4 * pi * area / perimeter^2 (完美圆=1.0)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    mean_val = 0
    if img_gray_roi is not None:
        mask = np.zeros(img_gray_roi.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(img_gray_roi, mask=mask)[0]

    return {
        "area": area,
        "circularity": circularity,
        "mean_val": mean_val,
        "contour": contour
    }

def watershed_segmentation_candidates(img_gray):
    """
    第一步：生成全图候选区域 (Candidate Generation)
    使用 距离变换 + 分水岭 算法，尽可能把粘连的细胞分开
    """
    # 1. 预处理
    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 2. 二值化 (Otsu)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 形态学去噪 (开运算)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. 距离变换寻找确定的前景 (Sure Foreground)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0) # 0.5 是距离阈值，可调
    sure_fg = np.uint8(sure_fg)

    # 5. 确定的背景 (Sure Background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 6. 未知区域 (Unknown)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. 分水岭标记
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # 8. 提取所有候选轮廓
    candidates = []
    # label 0 是边界, 1 是背景, >1 是物体
    unique_labels = np.unique(markers)
    for label in unique_labels:
        if label <= 1: 
            continue
        
        # 创建当前 label 的掩码
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # 取该 label 区域最大的轮廓
            c = max(cnts, key=cv2.contourArea)
            candidates.append(c)
            
    return candidates

def template_guided_detection(img_gray, roi_coords):
    """
    第二步：基于模板特征筛选 (Feature Matching)
    """
    # 1. 解析 ROI 并提取模板特征
    rx, ry, rw, rh = roi_coords['x'], roi_coords['y'], roi_coords['w'], roi_coords['h']
    
    # 确保 ROI 在图像范围内
    h, w = img_gray.shape
    rx, ry = max(0, rx), max(0, ry)
    rw, rh = min(w - rx, rw), min(h - ry, rh)
    
    roi_img = img_gray[ry:ry+rh, rx:rx+rw]
    
    # 对 ROI 做简单的 Otsu 获取主要轮廓特征
    _, roi_thresh = cv2.threshold(roi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi_cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not roi_cnts:
        return None, "ROI 中未检测到有效目标，请重新框选"
        
    template_cnt = max(roi_cnts, key=cv2.contourArea)
    tmpl_feats = get_contour_features(template_cnt, roi_img)
    
    # 2. 获取全图候选列表
    candidates = watershed_segmentation_candidates(img_gray)
    
    # 3. 筛选逻辑 (Matching Logic)
    final_buds = []
    
    # 设定容差 (Tolerance)
    area_tol = 0.50      # 面积允许 ±50% 差异
    circ_threshold = 0.6 # 圆度至少大于 0.6 (防止长条形噪声)
    
    for cnt in candidates:
        cand_feats = get_contour_features(cnt) # 这里为了速度暂时不传 img_gray 算灰度，只用几何特征
        
        # A. 面积筛选
        area_diff = abs(cand_feats['area'] - tmpl_feats['area']) / tmpl_feats['area']
        if area_diff > area_tol:
            continue
            
        # B. 圆度筛选 (Bud 应该是圆的)
        if cand_feats['circularity'] < circ_threshold:
            continue
            
        final_buds.append(cnt)
        
    return final_buds, tmpl_feats

# ==========================================
# 2. 辅助功能：Plotly ROI 解析与绘制
# ==========================================

def parse_plotly_relayout(relayout_data):
    """解析 Plotly 传回的框选数据，只取最后一个框"""
    if not relayout_data:
        return None
    
    # 处理 'shapes[0].x0' 这种扁平结构
    if "shapes[0].x0" in relayout_data:
        x0 = relayout_data["shapes[0].x0"]
        x1 = relayout_data["shapes[0].x1"]
        y0 = relayout_data["shapes[0].y0"]
        y1 = relayout_data["shapes[0].y1"]
    # 处理 'shapes': [{'x0':...}] 这种嵌套结构
    elif "shapes" in relayout_data and len(relayout_data["shapes"]) > 0:
        last_shape = relayout_data["shapes"][-1]
        x0, x1 = last_shape["x0"], last_shape["x1"]
        y0, y1 = last_shape["y0"], last_shape["y1"]
    else:
        return None

    return {
        "x": int(min(x0, x1)),
        "y": int(min(y0, y1)),
        "w": int(abs(x1 - x0)),
        "h": int(abs(y1 - y0))
    }

def reset_callback():
    """撤销 ROI 的回调"""
    st.session_state['roi_coords'] = None
    st.session_state['processed_result'] = None

# ==========================================
# 3. Streamlit UI 主逻辑
# ==========================================

st.title("🔬 智能细胞计数系统 (工程重构版)")
st.markdown("""
<style>
    .big-font { font-size:18px !important; }
    .result-box { border: 2px solid #ddd; padding: 15px; border-radius: 10px; }
</style>
此版本采用 **ROI 模板驱动 (Template-Guided)** 算法。
1. 上传图片 -> 2. 在图上框选**一个标准 Bud** -> 3. 算法自动寻找相似目标。
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 第一步：上传显微图像", type=["png", "jpg", "tif"])

if uploaded_file:
    # 加载图片
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    h, w, _ = img_array.shape
    
    # --- 交互区域 ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🖱️ 第二步：框选模板 (ROI)")
        
        # 创建 Plotly 图形
        fig = px.imshow(pil_img)
        fig.update_layout(
            dragmode='drawrect', # 激活画矩形模式
            newshape=dict(line_color='cyan', line_width=3),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500
        )
        
        # 如果 Session 中已经有 ROI，把它画出来 (持久化显示)
        if st.session_state['roi_coords']:
            rc = st.session_state['roi_coords']
            fig.add_shape(
                type="rect",
                x0=rc['x'], y0=rc['y'], x1=rc['x']+rc['w'], y1=rc['y']+rc['h'],
                line=dict(color="green", width=4),
            )
            fig.update_layout(title=dict(text="✅ 已锁定 ROI (如需重画请点击下方撤销)", font=dict(color="green")))

        # 渲染 Plotly
        # key 保持不变，通过 session state 管理数据
        relayout_data = st.plotly_chart(fig, use_container_width=True, on_select="ignore") 
        
        # --- 状态更新逻辑 ---
        # 只有当用户真的画了新框时，才更新 session_state
        # 注意：Streamlit 的 st.plotly_chart 返回值比较 tricky，需要判断 keys
        if relayout_data and ("shapes" in relayout_data or "shapes[0].x0" in relayout_data):
            new_roi = parse_plotly_relayout(relayout_data)
            if new_roi:
                st.session_state['roi_coords'] = new_roi
                st.rerun() # 强制刷新以锁定绿框

    with col2:
        st.subheader("⚙️ 操作面板")
        
        # 撤销按钮
        if st.session_state['roi_coords']:
            st.info(f"当前模板区域: \nX: {st.session_state['roi_coords']['x']}, Y: {st.session_state['roi_coords']['y']}")
            st.button("🔄 撤销 / 重画 ROI", on_click=reset_callback, type="primary")
            
            st.divider()
            
            # 开始识别按钮
            if st.button("🚀 开始识别分析"):
                with st.spinner("正在进行特征提取与全图匹配..."):
                    buds, template_features = template_guided_detection(img_gray, st.session_state['roi_coords'])
                    
                    if template_features and isinstance(template_features, dict):
                        # 绘制结果图
                        res_img = img_array.copy()
                        
                        # 1. 画 ROI (绿色)
                        rx, ry, rw, rh = st.session_state['roi_coords']['x'], st.session_state['roi_coords']['y'], st.session_state['roi_coords']['w'], st.session_state['roi_coords']['h']
                        cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                        
                        # 2. 画识别到的 Buds (红色细线) + 质心 (黄色点)
                        for cnt in buds:
                            # 轮廓
                            cv2.drawContours(res_img, [cnt], -1, (255, 0, 0), 2) 
                            # 质心
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                cv2.circle(res_img, (cX, cY), 3, (255, 255, 0), -1)

                        # 保存结果到 Session
                        st.session_state['processed_result'] = {
                            'image': res_img,
                            'count': len(buds),
                            'template_area': template_features['area']
                        }
                    else:
                        st.error(template_features) # 显示错误信息

        else:
            st.warning("👈 请先在左图框选一个标准 Bud")

    # --- 结果展示区 ---
    if st.session_state['processed_result']:
        st.divider()
        st.subheader("📊 分析结果")
        
        res_data = st.session_state['processed_result']
        
        # 显示统计指标
        m1, m2 = st.columns(2)
        m1.metric("识别总数 (Count)", f"{res_data['count']} 个")
        m2.metric("模板基准面积", f"{int(res_data['template_area'])} px²")
        
        # 显示结果大图
        # 限制显示宽度，优化观感
        st.image(
            res_data['image'], 
            caption=f"识别结果 visualization (Count: {res_data['count']}) - 黄点为质心，红线为轮廓", 
            width=800  # 限制最大显示宽度
        )

else:
    st.info("👋 欢迎使用。请上传图片开始。")
