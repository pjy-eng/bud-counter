import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="Click & Count", layout="wide")

# ==========================================
# 1. 核心算法：多点平均模板匹配
# ==========================================
def run_multi_point_matching(img_gray, points, params):
    h, w = img_gray.shape
    radius = params['radius'] # 采样半径
    window_size = radius * 2
    
    # --- A. 提取并合成模板 ---
    collected_patches = []
    
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        
        # 边界检查
        y1 = max(0, y - radius)
        y2 = min(h, y + radius)
        x1 = max(0, x - radius)
        x2 = min(w, x + radius)
        
        patch = img_gray[y1:y2, x1:x2]
        
        # 只有当截取的大小完全符合预期时才加入（避免边缘点尺寸不一）
        if patch.shape == (window_size, window_size):
            collected_patches.append(patch)
            
    if not collected_patches:
        return [], img_gray, np.zeros((10,10)), "无法提取有效模板，请不要点击图片边缘。"

    # **核心魔法**：计算平均模板 (Average Template)
    # 这能极大降低噪点，比单个框选更准
    avg_template = np.mean(collected_patches, axis=0).astype(np.uint8)
    
    # 简单的 CLAHE 增强 (对模板也做一下，保证特征清晰)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 注意：为了匹配，我们需要对原图做同样的增强
    img_enhanced = clahe.apply(img_gray)
    template_enhanced = clahe.apply(avg_template)

    # --- B. 匹配 ---
    res = cv2.matchTemplate(img_enhanced, template_enhanced, cv2.TM_CCOEFF_NORMED)
    
    # --- C. 筛选 ---
    loc = np.where(res >= params['threshold'])
    boxes = []
    
    for pt in zip(*loc[::-1]):
        boxes.append([int(pt[0]), int(pt[1]), window_size, window_size])
        
    # NMS 去重
    rects, _ = cv2.groupRectangles(boxes, groupThreshold=1, eps=0.3)
    
    # --- D. 绘图 ---
    res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_buds = []
    
    for (x, y, wb, hb) in rects:
        # 画红框
        cv2.rectangle(res_img, (x, y), (x + wb, y + hb), (0, 0, 255), 2)
        # 画个中心点
        cv2.circle(res_img, (x + wb//2, y + hb//2), 2, (0, 0, 255), -1)
        final_buds.append([x, y])

    # 把用户点击的点标成绿色，方便对比
    for pt in points:
        cv2.circle(res_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    return final_buds, res_img, avg_template, f"基于 {len(collected_patches)} 个样本生成的平均模板"

# ==========================================
# 2. UI 布局
# ==========================================
st.sidebar.header("🎛️ 设置")

# 关键参数：采样大小
radius = st.sidebar.slider("样本半径 (Radius)", 10, 60, 25, help="点击点周围多大区域算作一个 Bud？通常设为 Bud 直径的一半。")
threshold = st.sidebar.slider("相似度阈值", 0.3, 0.95, 0.55, help="越低找得越多。")

st.title("👆 点击即计数 (Click & Count)")
st.caption("操作方式：不要画框，直接在左图中 **点击** 几个你认为是 Bud 的目标。")

uploaded_file = st.file_uploader("上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1. 点击样本 (Click Points)")
        # 使用 Point 模式
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 1)",
            stroke_color="#00FF00",
            background_image=pil_img,
            update_streamlit=True,
            height=500,
            drawing_mode="point", # 关键：点选模式
            point_display_radius=6,
            key="canvas_click"
        )
        # 清除按钮提示
        if st.button("🗑️ 清除所有点击"):
            st.rerun()

    with col2:
        st.subheader("2. 分析结果")
        
        # 获取点击点
        points = []
        if canvas.json_data and len(canvas.json_data["objects"]) > 0:
            for obj in canvas.json_data["objects"]:
                # 获取圆心坐标
                cx = obj['left']
                cy = obj['top']
                points.append([cx, cy])
        
        if len(points) > 0:
            params = {'radius': radius, 'threshold': threshold}
            
            with st.spinner("正在合成特征并搜索..."):
                buds, res_img, template, msg = run_multi_point_matching(img_gray, points, params)
            
            # 显示计数
            st.metric("✅ 找到目标", f"{len(buds)} 个")
            
            # 显示合成的模板 (让用户知道AI学到了什么)
            st.write("🧠 AI 学习到的平均特征:")
            st.image(template, width=100, clamp=True, channels='GRAY')
            
            # 显示大图
            fig = px.imshow(res_img)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👈 请在左图点击至少 1 个 Bud。建议点击 3 个以上以获得最佳效果。")

else:
    st.info("请上传图片。")
