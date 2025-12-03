import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="Click on Image", layout="wide")

# ==========================================
# 1. 核心算法：多点平均模板匹配
# ==========================================
def run_multi_point_matching(img_gray, points, params):
    h, w = img_gray.shape
    radius = params['radius']
    window_size = radius * 2
    
    collected_patches = []
    
    # 提取点击点周围的图像块
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        y1, y2 = max(0, y - radius), min(h, y + radius)
        x1, x2 = max(0, x - radius), min(w, x + radius)
        patch = img_gray[y1:y2, x1:x2]
        
        if patch.shape == (window_size, window_size):
            collected_patches.append(patch)
            
    if not collected_patches:
        return [], img_gray, np.zeros((10,10)), "无法提取有效模板，请勿点击边缘。"

    # 计算平均模板
    avg_template = np.mean(collected_patches, axis=0).astype(np.uint8)
    
    # 预处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    template_enhanced = clahe.apply(avg_template)

    # 匹配
    res = cv2.matchTemplate(img_enhanced, template_enhanced, cv2.TM_CCOEFF_NORMED)
    
    # 筛选
    loc = np.where(res >= params['threshold'])
    boxes = []
    for pt in zip(*loc[::-1]):
        boxes.append([int(pt[0]), int(pt[1]), window_size, window_size])
        
    rects, _ = cv2.groupRectangles(boxes, groupThreshold=1, eps=0.3)
    
    # 绘图
    res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_buds = []
    
    for (x, y, wb, hb) in rects:
        cv2.rectangle(res_img, (x, y), (x + wb, y + hb), (0, 0, 255), 2)
        final_buds.append([x, y])

    for pt in points:
        cv2.circle(res_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    return final_buds, res_img, avg_template, ""

# ==========================================
# 2. UI 布局
# ==========================================
st.title("👆 直接点击版 (Click on Image)")
st.caption("现在图片会自动铺满区域，没有白色背景了。")

# 侧边栏参数
st.sidebar.header("🎛️ 参数")
# 注意：因为图片可能被缩放显示，这里的半径需要根据视觉大小来调，默认给小一点
radius = st.sidebar.slider("样本半径 (Radius)", 10, 50, 20)
threshold = st.sidebar.slider("相似度阈值", 0.3, 0.95, 0.60)

uploaded_file = st.file_uploader("上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    # 1. 加载原始图片
    pil_img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = pil_img.size
    
    # 2. 关键步骤：计算适应屏幕的显示尺寸
    # 为了防止图片太大撑破屏幕，或者太小留白，我们将宽度固定为适宜大小（如 700px）
    # 并保持长宽比缩放
    display_width = 700
    ratio = display_width / orig_w
    display_height = int(orig_h * ratio)
    
    # 缩放图片用于 Canvas 显示和处理 (这样速度也更快)
    pil_img_resized = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
    img_array = np.array(pil_img_resized)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("1. 点击样本")
        
        # 3. Canvas 设置：宽高完全等于缩放后的图片宽高
        # 这样就没有白色背景了！
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 1)",
            stroke_color="#00FF00",
            background_image=pil_img_resized, # 使用缩放后的图做背景
            update_streamlit=True,
            width=display_width,   # 强制宽度
            height=display_height, # 强制高度
            drawing_mode="point",
            point_display_radius=5,
            key="canvas_immersive"
        )
        
        if st.button("🗑️ 清除点击"):
            st.rerun()

    with col2:
        st.subheader("2. 结果")
        
        points = []
        if canvas.json_data and len(canvas.json_data["objects"]) > 0:
            for obj in canvas.json_data["objects"]:
                points.append([obj['left'], obj['top']])
        
        if len(points) > 0:
            params = {'radius': radius, 'threshold': threshold}
            
            with st.spinner("分析中..."):
                buds, res_img, template, _ = run_multi_point_matching(img_gray, points, params)
            
            st.metric("✅ 计数", f"{len(buds)} 个")
            
            # 显示合成模板
            st.write("平均特征:")
            st.image(template, width=80, clamp=True, channels='GRAY')
            
            # 显示结果
            st.image(res_img, use_column_width=True, caption="红框=AI找到的目标")
        else:
            st.info("👈 请在左图直接点击 Bud 中心。")

else:
    st.info("请上传图片。")
