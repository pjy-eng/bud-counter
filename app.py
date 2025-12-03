import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Bud Counter Toolbox", layout="wide")

# ==========================================
# 算法引擎 1: 模板匹配 (您之前觉得好用的版本)
# ==========================================
def run_template_matching(img_gray, roi_coords, threshold):
    # 1. 预处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    
    # 2. 提取模板
    rx, ry, rw, rh = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
    if rw <= 0 or rh <= 0: return [], img_enhanced, "模板无效"
    
    template = img_enhanced[ry:ry+rh, rx:rx+rw]
    
    # 3. 匹配
    res = cv2.matchTemplate(img_enhanced, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    
    # 4. 转换结果
    boxes = []
    for pt in zip(*loc[::-1]):
        boxes.append([int(pt[0]), int(pt[1]), rw, rh])
    
    # 5. 去重
    rects, _ = cv2.groupRectangles(boxes, groupThreshold=1, eps=0.3)
    
    # 6. 绘图
    res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_buds = []
    user_center = (rx + rw//2, ry + rh//2)
    
    for (x, y, w, h) in rects:
        # 排除用户自己画的那个
        curr_center = (x + w//2, y + h//2)
        dist = np.sqrt((user_center[0]-curr_center[0])**2 + (user_center[1]-curr_center[1])**2)
        if dist < rw / 2: continue
            
        final_buds.append([x, y, w, h])
        cv2.rectangle(res_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
    return final_buds, res_img, None

# ==========================================
# 算法引擎 2: 霍夫圆变换 (新方法)
# ==========================================
def run_hough_circles(img_gray, params):
    # 1. 预处理 (霍夫变换对噪点极其敏感，需要强力模糊)
    # 中值模糊去除椒盐噪点
    blurred = cv2.medianBlur(img_gray, 5)
    
    # 2. 霍夫圆检测
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1,                   # 分辨率倒数
        minDist=params['min_dist'], # 圆心之间的最小距离
        param1=params['canny_th'],  # Canny 边缘检测的高阈值
        param2=params['accum_th'],  # 圆心累加器阈值 (越小越容易检测到圆，也容易误检)
        minRadius=params['min_r'], 
        maxRadius=params['max_r']
    )
    
    res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    buds = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # i = [x, y, r]
            buds.append(i)
            # 画圆
            cv2.circle(res_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 画圆心
            cv2.circle(res_img, (i[0], i[1]), 2, (0, 0, 255), 3)
            
    return buds, res_img

# ==========================================
# 主界面逻辑
# ==========================================
st.title("🔬 细胞计数工具箱 (双引擎版)")

# 侧边栏：选择算法
algorithm = st.sidebar.selectbox("🛠️ 选择核心算法", ["A. 模板匹配 (纹理)", "B. 霍夫圆检测 (几何形状)"])

uploaded_file = st.file_uploader("上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    col1, col2 = st.columns([2, 1])

    # ================= 模式 A: 模板匹配 =================
    if algorithm == "A. 模板匹配 (纹理)":
        st.sidebar.divider()
        st.sidebar.markdown("### A 模式参数")
        tm_thresh = st.sidebar.slider("相似度阈值", 0.3, 0.95, 0.60, 0.01)
        
        with col1:
            st.subheader("1. 框选模板")
            canvas = st_canvas(
                fill_color="rgba(0, 255, 0, 0.2)",
                stroke_color="#00FF00",
                background_image=pil_img,
                update_streamlit=True,
                height=500,
                drawing_mode="rect",
                key="canvas_a"
            )
            
        with col2:
            st.subheader("2. 分析结果")
            if canvas.json_data and len(canvas.json_data["objects"]) > 0:
                obj = canvas.json_data["objects"][-1]
                roi = {'left':int(obj['left']), 'top':int(obj['top']), 'width':int(obj['width']), 'height':int(obj['height'])}
                
                if roi['width'] > 0:
                    buds, res_img, _ = run_template_matching(img_gray, roi, tm_thresh)
                    st.metric("计数 (含模板)", f"{len(buds)+1} 个")
                    st.image(res_img, use_column_width=True)
            else:
                st.info("请在左侧画框。")

    # ================= 模式 B: 霍夫圆检测 =================
    else:
        st.sidebar.divider()
        st.sidebar.markdown("### B 模式参数 (霍夫圆)")
        # 霍夫变换的参数比较多，这里提供最关键的调节
        h_min_dist = st.sidebar.slider("圆心最小间距 (minDist)", 10, 100, 30, help="如果结果重叠严重，调大此值")
        h_accum_th = st.sidebar.slider("检测灵敏度 (Accumulator)", 10, 100, 30, help="越小越灵敏(圆越多)，越大越严格")
        h_min_r = st.sidebar.slider("最小半径", 5, 50, 15)
        h_max_r = st.sidebar.slider("最大半径", 20, 100, 50)
        
        with col1:
            st.subheader("1. 原始图像")
            st.image(pil_img, use_column_width=True)
            st.caption("霍夫变换不需要画框，它会自动全图找圆。")
            
        with col2:
            st.subheader("2. 自动检圆结果")
            # 实时计算
            params = {
                'min_dist': h_min_dist, 'canny_th': 100, 
                'accum_th': h_accum_th, 'min_r': h_min_r, 'max_r': h_max_r
            }
            buds, res_img = run_hough_circles(img_gray, params)
            
            st.metric("检测到的圆", f"{len(buds)} 个")
            st.image(res_img, use_column_width=True)
            
            if len(buds) == 0:
                st.warning("未检测到圆。请尝试降低'检测灵敏度'数值，或调整半径范围。")

else:
    st.info("请先上传图片。")
