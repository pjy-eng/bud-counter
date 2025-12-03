import streamlit as st
from PIL import Image
import numpy as np
# 引入画布组件
from streamlit_drawable_canvas import st_canvas
import draw_buds as db

# 页面配置
st.set_page_config(page_title="交互式细胞出芽检测", page_icon="🔬", layout="wide")

st.title("🔬 交互式细胞出芽检测系统")
st.markdown("---")

# --- 侧边栏：控制面板 ---
with st.sidebar:
    st.header("1. 图像与设置")
    uploaded_file = st.file_uploader("上传显微镜图片:", type=["png", "jpg", "jpeg", "tif"])
    
    st.markdown("---")
    st.header("2. 检测参数")
    # 添加一个滑动条来调整匹配的严格程度
    threshold = st.slider(
        "相似度阈值 (Threshold)", 
        min_value=0.5, max_value=0.95, value=0.75, step=0.05,
        help="值越高，匹配越严格，漏检可能增加；值越低，误检可能增加。"
    )
    
    run_button = st.button("开始寻找相似目标 🚀", type="primary", disabled=not uploaded_file)

    st.markdown("---")
    st.markdown("""
    ### 📖 操作指南:
    1. 上传图片。
    2. 在右侧画布上，用鼠标**框选**一个或多个典型的“芽”。
    3. 调整相似度阈值（可选）。
    4. 点击“开始寻找”按钮。
    """)

# --- 主界面 ---
if uploaded_file:
    # 加载原始图片
    pil_image = Image.open(uploaded_file)
    # 获取图片尺寸，用于设置画布大小
    img_w, img_h = pil_image.size
    # 计算显示时的缩放比例，防止图片过大撑破布局 (最大宽度设置为 700)
    display_width = min(700, img_w)
    scale_factor = display_width / img_w
    display_height = int(img_h * scale_factor)

    # 创建两列布局
    col_canvas, col_result = st.columns(2)
    
    with col_canvas:
        st.subheader("👉 请在此处框选样本 (Draw Here)")
        # --- 核心组件：可绘图画布 ---
        # 将图片作为画布背景，允许用户画矩形 (rect)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # 填充色 (半透明橙色)
            stroke_color="#FF0000",              # 边框色 (红色)
            stroke_width=2,
            background_image=pil_image,          # 设置背景图
            update_streamlit=True,               # 实时更新
            height=display_height,               # 设置画布显示高度
            width=display_width,                 # 设置画布显示宽度
            drawing_mode="rect",                 # 模式：画矩形
            key="canvas",
        )

    # --- 处理逻辑 ---
    with col_result:
        st.subheader("检测结果 (Result)")
        result_container = st.empty() # 创建一个空容器用于占位

        if run_button:
            # 检查用户是否画了框
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                with st.spinner("正在努力寻找相似的芽..."):
                    # 1. 获取用户画的所有矩形框数据
                    objects = canvas_result.json_data["objects"]
                    user_boxes = []
                    for obj in objects:
                        # 将画布上的显示坐标还原为图片的真实坐标
                        user_boxes.append({
                            'left': obj['left'] / scale_factor,
                            'top': obj['top'] / scale_factor,
                            'width': obj['width'] / scale_factor,
                            'height': obj['height'] / scale_factor
                        })

                    # 2. 调用后端函数进行搜索
                    result_image, count = db.find_similar_buds(pil_image, user_boxes, threshold)
                    
                    # 3. 显示结果
                    result_container.image(result_image, use_column_width=True)
                    
                    if count > 0:
                        st.success(f"🎉 成功找到 {count} 个相似目标！")
                    else:
                        st.warning("未找到相似目标，请尝试降低阈值或重新圈选更清晰的样本。")
            else:
                result_container.info("请先在左侧图片上至少框选一个样本。")
else:
    st.info("👈 请先在侧边栏上传一张图片。")
