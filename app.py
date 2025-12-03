import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import time

# 尝试导入 Cellpose，如果环境没装好会报错
try:
    from cellpose import models, utils
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

st.set_page_config(page_title="AI Bud Counter (Cellpose)", layout="wide")

# ==========================================
# AI 核心逻辑
# ==========================================
@st.cache_resource
def load_cellpose_model():
    """
    加载模型只做一次，并缓存起来，防止每次点击都重新加载
    """
    # model_type='cyto' 是通用的细胞模型
    # gpu=False 表示强制使用 CPU (Streamlit Cloud 没有 GPU)
    print("⏳ 正在下载/加载 Cellpose 模型...")
    model = models.Cellpose(model_type='cyto', gpu=False)
    return model

def run_ai_prediction(img_rgb, diameter, flow_threshold, cellprob_threshold):
    # 加载模型
    model = load_cellpose_model()
    
    # 开始预测
    # channels=[0,0] 表示灰度图或自动推断
    # diameter: 细胞大概的直径
    masks, flows, styles, diams = model.eval(
        img_rgb, 
        diameter=diameter,
        channels=[0,0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    
    return masks

# ==========================================
# UI 布局
# ==========================================
st.title("🤖 AI 细胞计数 (Cellpose 云端版)")

if not CELLPOSE_AVAILABLE:
    st.error("❌ 未检测到 Cellpose 库。请检查 requirements.txt 是否包含了 'cellpose'。")
    st.stop()

st.info("💡 提示：这是一个深度学习模型。在云端 CPU 上运行可能需要 10~30 秒，请耐心等待。")

# --- 侧边栏：AI 参数 ---
st.sidebar.header("🧠 AI 参数设置")

# 直径 (Diameter) 是最重要的参数
diameter = st.sidebar.number_input(
    "预估 Bud 直径 (像素)", 
    min_value=10, max_value=200, value=60, step=5,
    help="大概估算一下你的 Bud 有多大。如果设为 0，AI 会尝试自动估算（更慢）。"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 进阶微调")
flow_th = st.sidebar.slider("形态一致性 (Flow Thresh)", 0.0, 1.0, 0.4, 0.1, help="值越小，要求形状越规则；值越大，允许更多异形。")
cellprob_th = st.sidebar.slider("置信度 (Cell Prob)", -6.0, 6.0, 0.0, 0.5, help="值越低，找到的越多（可能误检）；值越高，越严格。")

# --- 主界面 ---
uploaded_file = st.file_uploader("上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. 原始图像")
        st.image(pil_img, use_column_width=True)

    with col2:
        st.subheader("2. AI 分析结果")
        
        # 添加一个大按钮来触发计算，避免自动运行太卡
        if st.button("🚀 运行 AI 分析", type="primary"):
            with st.spinner("AI 正在思考中... (可能需要几十秒)"):
                start_time = time.time()
                
                # 运行预测
                masks = run_ai_prediction(img_array, diameter, flow_th, cellprob_th)
                
                # 处理结果
                num_cells = masks.max()
                end_time = time.time()
                
                # 绘制轮廓
                # 获取轮廓线条
                outlines = utils.outlines_list(masks)
                
                res_img = img_array.copy()
                for o in outlines:
                    # o 是 [y, x] 坐标
                    pts = o.reshape((-1, 1, 2)).astype(np.int32)
                    # 注意 cellpose 返回的是 y,x，opencv 需要 x,y，需要翻转一下
                    # utils.outlines_list 返回的通常已经是像素坐标，但顺序可能需要调整
                    # 这里直接用 matplotlib 的思路画图可能不方便，我们用 cv2 画
                    # 需要把 [y, x] 转为 [x, y]
                    pts_xy = np.flip(pts, axis=2) 
                    cv2.polylines(res_img, [pts_xy], isClosed=True, color=(255, 0, 0), thickness=2)

                st.success(f"✅ 识别完成！找到 {num_cells} 个目标 (耗时 {end_time-start_time:.1f}s)")
                st.image(res_img, caption=f"Count: {num_cells}", use_column_width=True)
                
else:
    st.info("请上传图片后点击运行。")
