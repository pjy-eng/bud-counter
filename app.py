import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px

# ==========================================
# 0. 全局配置与状态初始化
# ==========================================
st.set_page_config(page_title="High-Precision Bud Counter", layout="wide")

# 初始化 Session State
if 'roi_coords' not in st.session_state:
    st.session_state['roi_coords'] = None

# ==========================================
# 1. 核心算法库 (包含防崩溃修复)
# ==========================================

def get_contour_features(contour):
    """计算轮廓的面积和圆度"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    return {"area": area, "circularity": circularity}

def process_and_count(img_gray, roi_coords, params):
    """
    核心处理流程 - 修复了类型错误和空值崩溃问题
    """
    try:
        # --- A. 预处理与类型安全检查 ---
        # 强制转换为 uint8，防止 OpenCV 报错
        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8)

        # CLAHE 增强对比度
        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # --- B. 图像分割 ---
        # 1. Otsu 二值化
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 形态学去噪 (确保 kernel size 是合法的奇数)
        k_size = int(params['open_kernel'])
        if k_size % 2 == 0: k_size += 1 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # --- C. 分水岭算法 (分离粘连) ---
        # 确定的背景
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定的前景 (距离变换)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # 安全检查：防止图片全黑导致 max() 为 0
        dist_max = dist_transform.max()
        if dist_max == 0:
            return None, None, "图像预处理后为空，请降低'去噪强度'或调整对比度。"

        _, sure_fg = cv2.threshold(dist_transform, params['dist_ratio'] * dist_max, 255, 0)
        sure_fg = np.uint8(sure_fg) # 关键：必须转为 uint8
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 连通域标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        # --- D. 提取模板特征 ---
        rx, ry, rw, rh = roi_coords['x'], roi_coords['y'], roi_coords['w'], roi_coords['h']
        
        # 边界检查
        h, w = img_gray.shape
        rx, ry = max(0, rx), max(0, ry)
        rw = min(w - rx, rw)
        rh = min(h - ry, rh)
        
        if rw <= 0 or rh <= 0:
            return None, None, "ROI 区域无效，请重新框选。"

        # 尝试从处理后的图提取特征（更准）
        roi_region_bin = opening[ry:ry+rh, rx:rx+rw]
        roi_cnts, _ = cv2.findContours(roi_region_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果处理后的图中 ROI 空了，回退到用原图 Otsu 提取
        if not roi_cnts:
            roi_raw = img_gray[ry:ry+rh, rx:rx+rw]
            _, roi_backup_thresh = cv2.threshold(roi_raw, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            roi_cnts, _ = cv2.findContours(roi_backup_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not roi_cnts:
            return None, None, "模板区域内未检测到目标，请选择更清晰的 Bud。"
            
        template_cnt = max(roi_cnts, key=cv2.contourArea)
        tmpl_feats = get_contour_features(template_cnt)

        # --- E. 全图匹配筛选 ---
        final_buds = []
        
        # 遍历 marker (从2开始，0是边界，1是背景)
        unique_markers = np.unique(markers)
        for label in unique_markers:
            if label <= 1: continue 
            
            mask = np.zeros(img_gray.shape, dtype=np.uint8)
            mask[markers == label] = 255
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            
            c = max(cnts, key=cv2.contourArea)
            feats = get_contour_features(c)
            
            # 1. 面积筛选
            # 加上极小值防止除零
            area_ratio = abs(feats['area'] - tmpl_feats['area']) / (tmpl_feats['area'] + 1e-5)
            if area_ratio > params['area_tol']:
                continue 
                
            # 2. 圆度筛选
            if feats['circularity'] < params['circ_thresh']:
                continue 
                
            final_buds.append(c)

        # --- F. 绘图结果 ---
        res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        # 绘制所有 Bud (红色)
        cv2.drawContours(res_img, final_buds, -1, (0, 0, 255), 2) 
        
        # 绘制 ROI (绿色)
        cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        
        # 绘制质心 (黄色点)
        for c in final_buds:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(res_img, (cX, cY), 3, (0, 255, 255), -1)

        return final_buds, res_img, tmpl_feats

    except Exception as e:
        return None, None, f"算法内部错误: {str(e)}"

# ==========================================
# 2. 辅助函数
# ==========================================

def parse_plotly_relayout(relayout_data):
    """解析 Plotly 框选数据"""
    if not relayout_data: return None
    # 兼容两种 Plotly 数据格式
    if "shapes[0].x0" in relayout_data:
        x0, x1 = relayout_data["shapes[0].x0"], relayout_data["shapes[0].x1"]
        y0, y1 = relayout_data["shapes[0].y0"], relayout_data["shapes[0].y1"]
    elif "shapes" in relayout_data and len(relayout_data["shapes"]) > 0:
        last = relayout_data["shapes"][-1]
        x0, x1, y0, y1 = last["x0"], last["x1"], last["y0"], last["y1"]
    else: return None
    
    return {
        "x": int(min(x0, x1)), "y": int(min(y0, y1)), 
        "w": int(abs(x1 - x0)), "h": int(abs(y1 - y0))
    }

def reset_callback():
    """重置状态"""
    st.session_state['roi_coords'] = None

# ==========================================
# 3. Streamlit UI 布局
# ==========================================

# --- 侧边栏：参数微调 ---
st.sidebar.header("🎛️ 算法参数微调")
st.sidebar.markdown("通过调整这些参数来逼近 90% 准确率")

params = {
    'clahe_clip': st.sidebar.slider("对比度增强 (CLAHE)", 1.0, 5.0, 2.0, 0.5, help="增加对比度，让模糊的细胞边界更清晰"),
    'open_kernel': st.sidebar.slider("去噪强度 (Kernel)", 1, 9, 3, 2, help="值越大，去除的小噪点越多，但可能丢失小 Bud"),
    'dist_ratio': st.sidebar.slider("粘连分离灵敏度", 0.1, 0.9, 0.5, 0.05, help="决定分水岭的前景范围。越小分得越细，越大越容易粘连"),
    'area_tol': st.sidebar.slider("面积容差 (±%)", 0.1, 1.5, 0.5, 0.05, help="允许目标大小与模板差异的程度。0.5代表允许±50%"),
    'circ_thresh': st.sidebar.slider("最小圆度限制", 0.1, 1.0, 0.6, 0.05, help="越接近1越圆。调高此值可过滤长条形背景杂质")
}

st.title("🔬 Pro 级细胞 Bud 计数系统")

uploaded_file = st.file_uploader("1. 上传图片", type=["jpg", "png", "tif"])

if uploaded_file:
    # 加载图片
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # 布局
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("2. 框选模板 ROI")
        # 创建 Plotly 画布
        fig = px.imshow(pil_img)
        fig.update_layout(
            dragmode='drawrect', 
            newshape=dict(line_color='cyan', line_width=3), 
            height=550, 
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        # 如果已锁定 ROI，画出绿框
        if st.session_state['roi_coords']:
            r = st.session_state['roi_coords']
            fig.add_shape(type="rect", x0=r['x'], y0=r['y'], x1=r['x']+r['w'], y1=r['y']+r['h'], line=dict(color="green", width=4))
        
        # 渲染图表并获取交互数据
        relayout_data = st.plotly_chart(fig, use_container_width=True, on_select="ignore")
        
        # 检测用户画图动作
        if relayout_data and ("shapes" in relayout_data or "shapes[0].x0" in relayout_data):
            new_roi = parse_plotly_relayout(relayout_data)
            # 只有当确实画了新框，且与旧框不同时才刷新
            if new_roi and new_roi != st.session_state['roi_coords']:
                st.session_state['roi_coords'] = new_roi
                st.rerun()

    with col2:
        st.subheader("3. 结果分析")
        
        if st.session_state['roi_coords']:
            # 撤销按钮
            st.button("🔄 撤销 / 重画 ROI", on_click=reset_callback)
            
            st.divider()
            
            # 实时计算
            with st.spinner("计算中..."):
                buds, res_img, tmpl_feats_or_msg = process_and_count(img_gray, st.session_state['roi_coords'], params)
            
            # 结果显示逻辑
            if buds is not None:
                st.success(f"✅ 计数结果: **{len(buds)}** 个")
                
                # 显示结果图
                st.image(res_img, caption="红线=轮廓, 黄点=质心", use_column_width=True)
                
                # 显示调试信息
                st.info(f"""
                **模板特征:**
                - 面积: {int(tmpl_feats_or_msg['area'])} px
                - 圆度: {tmpl_feats_or_msg['circularity']:.2f}
                """)
            else:
                # 显示错误信息 (tmpl_feats_or_msg 在出错时存储错误文本)
                st.error(f"⚠️ {tmpl_feats_or_msg}")
        else:
            st.info("👈 请在左侧图像上框选一个标准的 Bud 作为模板。")

else:
    st.info("👋 请上传显微图像开始工作。")
