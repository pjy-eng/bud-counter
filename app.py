import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="Bud Counter (Template Match)", layout="wide")

if 'roi_coords' not in st.session_state:
    st.session_state['roi_coords'] = None

# ==========================================
# 1. 核心算法：模板匹配 (复现第二张图的逻辑)
# ==========================================
def process_with_template_matching(img_gray, roi_coords, params):
    try:
        # --- A. 预处理 ---
        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8)

        # 简单的 CLAHE 增强，和之前保持一致
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_gray)
        
        # --- B. 提取模板 ---
        # Canvas 坐标
        rx, ry, rw, rh = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
        
        # 边界检查
        h, w = img_enhanced.shape
        if rw <= 0 or rh <= 0 or rx >= w or ry >= h:
            return None, None, "框选区域无效。"
            
        # 裁剪模板
        template = img_enhanced[ry:ry+rh, rx:rx+rw]
        
        if template.shape[0] == 0 or template.shape[1] == 0:
            return None, None, "模板为空。"

        # --- C. 核心：matchTemplate (像素级匹配) ---
        # 使用归一化相关系数匹配法 (TM_CCOEFF_NORMED)
        # 这是最稳健的方法，结果在 0~1 之间
        res = cv2.matchTemplate(img_enhanced, template, cv2.TM_CCOEFF_NORMED)
        
        # --- D. 筛选与去重 (NMS) ---
        # 获取滑块设定的阈值
        threshold = params['match_thresh']
        
        # 找到所有大于阈值的点
        loc = np.where(res >= threshold)
        
        # 转换成矩形框列表 [x, y, w, h]
        boxes = []
        for pt in zip(*loc[::-1]):
            boxes.append([int(pt[0]), int(pt[1]), rw, rh])
            
        # 使用 OpenCV 的 groupRectangles 进行去重 (Non-Maximum Suppression)
        # groupThreshold=1 表示至少要有1次重叠才算有效（去噪）
        # eps=0.3 表示允许重叠的程度
        rects, weights = cv2.groupRectangles(boxes, groupThreshold=1, eps=0.3)
        
        # --- E. 绘图 ---
        res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        final_buds = []
        
        # 为了避免把自己画的那个框也算进去重复计数，我们需要计算距离
        user_center = (rx + rw//2, ry + rh//2)
        
        for (x, y, w_box, h_box) in rects:
            # 计算当前框的中心
            curr_center = (x + w_box//2, y + h_box//2)
            dist = np.sqrt((user_center[0]-curr_center[0])**2 + (user_center[1]-curr_center[1])**2)
            
            # 如果距离太近（比如小于模板宽度的一半），说明是用户自己画的那个，跳过
            if dist < rw / 2:
                continue
                
            final_buds.append([x, y, w_box, h_box])
            # 画红框
            cv2.rectangle(res_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

        # 画用户选的绿框
        cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        
        return final_buds, res_img, f"阈值: {threshold}"

    except Exception as e:
        return None, None, f"算法错误: {str(e)}"

# ==========================================
# 2. UI 布局
# ==========================================
st.sidebar.header("🎛️ 匹配参数")
st.sidebar.info("现在的算法逻辑是：'长得像的就圈出来'，不再受形状限制。")

params = {
    # 这是最重要的参数
    'match_thresh': st.sidebar.slider("相似度阈值 (Threshold)", 0.3, 0.95, 0.60, 0.01, 
                                    help="值越低，找出来的越多（但也可能找错）；值越高，越严格。")
}

st.title("🔬 Bud 计数器 (模板匹配版)")
st.caption("复刻 'Image 2' 的算法逻辑：基于纹理的像素匹配。")

uploaded_file = st.file_uploader("1. 上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("2. 框选模板")
        st.caption("请画一个框，框住一个标准的 Bud。")
        
        # 使用 Canvas (必须保留，用于稳定交互)
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_color="#00FF00",
            background_image=pil_img,
            update_streamlit=True,
            height=500, 
            drawing_mode="rect",
            key="canvas_tm", # 改个key防止缓存
        )

    with col2:
        st.subheader("3. 结果")
        
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            roi_coords = {
                'left': int(obj['left']), 'top': int(obj['top']),
                'width': int(obj['width']), 'height': int(obj['height'])
            }
            
            if roi_coords['width'] > 0:
                with st.spinner("正在进行全图扫描匹配..."):
                    # 调用新的模板匹配算法
                    buds, res_img, msg = process_with_template_matching(img_gray, roi_coords, params)

                if buds is not None:
                    # 总数 = 找到的相似 + 用户选的1个
                    total = len(buds) + 1
                    st.metric("✅ 总计数 (包含模板)", f"{total} 个")
                    
                    fig = px.imshow(res_img)
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(msg)
        else:
            st.info("👈 请先画框。")
else:
    st.info("👋 请先上传图片。")
