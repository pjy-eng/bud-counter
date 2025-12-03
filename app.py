import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import draw_buds as db

st.set_page_config(page_title="交互式 AI 细胞识别", page_icon="🧠", layout="wide")

st.title("🧠 交互式 AI 细胞出芽识别系统 (Human-in-the-Loop)")
st.markdown("""
这是一个**人机协作**系统。请在图片上**圈选一个标准的“芽”作为样本**，AI 大脑 (Gemini 2.0) 将会学习你的样本，并在整张图中寻找相似的目标。
""")
st.markdown("---")

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 配置与上传")
    # 安全输入 API Key
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    
    api_key_input = st.text_input("Google Gemini API Key:", type="password", value=st.session_state.api_key, help="需要使用 Gemini 2.0 Flash 模型")
    if api_key_input:
        st.session_state.api_key = api_key_input

    uploaded_file = st.file_uploader("上传显微镜图片...", type=['jpg', 'png', 'jpeg', 'tif'])
    
    st.divider()
    st.header("2. 操作指南")
    st.markdown("""
    1. 输入你的 API Key。
    2. 上传图片。
    3. 在右侧画布上，用鼠标**精确框选**一个你认为最标准的“芽”。
    4. 点击下方的“开始 AI 搜索”按钮。
    """)
    
    # 运行按钮
    run_button = st.button("开始 AI 搜索 (Start Search) 🚀", type="primary", disabled=not (uploaded_file and st.session_state.api_key))

# --- 主界面 ---
if uploaded_file and st.session_state.api_key:
    # 加载和调整图片
    pil_image = Image.open(uploaded_file)
    img_w, img_h = pil_image.size
    display_width = min(700, img_w)
    scale_factor = display_width / img_w
    display_height = int(img_h * scale_factor)

    # 创建两列布局
    col_canvas, col_result = st.columns(2)
    
    with col_canvas:
        st.subheader("👉 请在此处框选样本 (Draw Sample)")
        # 画布组件
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",  # 填充色 (半透明绿色)
            stroke_color="#00FF00",              # 边框色 (绿色)
            stroke_width=2,
            background_image=pil_image,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",                 # 模式：画矩形
            key="canvas_ai_teaching",
        )

    # --- 处理逻辑 ---
    with col_result:
        st.subheader("AI 识别结果 (Result)")
        result_container = st.empty()

        if run_button:
            # 检查用户是否画了框
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                with st.spinner("Gemini 正在学习你的样本并进行全图搜索..."):
                    # 1. 获取用户画的框，并还原坐标
                    objects = canvas_result.json_data["objects"]
                    user_boxes = []
                    for obj in objects:
                        user_boxes.append({
                            'left': obj['left'] / scale_factor,
                            'top': obj['top'] / scale_factor,
                            'width': obj['width'] / scale_factor,
                            'height': obj['height'] / scale_factor
                        })

                    # 2. 调用 AI 后端
                    result_image, count, error = db.detect_similar_buds(st.session_state.api_key, pil_image, user_boxes)
                    
                    # 3. 显示结果
                    if error:
                        st.error(f"发生错误: {error}")
                    else:
                        result_container.image(result_image, use_column_width=True)
                        st.success(f"✅ 分析完成！基于你的样本，Gemini 找到了 {count} 个相似目标。")
                        
            else:
                st.warning("⚠️ 请先在左侧图片上至少框选一个样本，然后再点击开始按钮。")
elif not st.session_state.api_key:
     st.info("👈 请先在侧边栏输入你的 Google Gemini API Key。")
else:
    st.info("👈 请先在侧边栏上传一张图片。")
