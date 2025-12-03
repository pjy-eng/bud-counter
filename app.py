import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import draw_buds as db

st.set_page_config(page_title="交互式 AI 细胞识别", page_icon="🧠", layout="wide")

st.title("🧠 交互式 AI 细胞出芽识别系统 (Human-in-the-Loop)")
st.markdown("---")

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 配置与上传")
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    
    api_key_input = st.text_input("Google Gemini API Key:", type="password", value=st.session_state.api_key)
    if api_key_input:
        st.session_state.api_key = api_key_input

    uploaded_file = st.file_uploader("上传显微镜图片...", type=['jpg', 'png', 'jpeg', 'tif'])
    
    st.divider()
    st.markdown("### 📖 操作指南")
    st.info("""
    如果下方的画布背景是空白的：
    请参考上方的“对照参考图”，在下方的空白画布上的**对应位置**，尽可能准确地框选一个“芽”。
    """)
    
    run_button = st.button("开始 AI 搜索 (Start Search) 🚀", type="primary", disabled=not (uploaded_file and st.session_state.api_key))

# --- 主界面 ---
if uploaded_file and st.session_state.api_key:
    pil_image = Image.open(uploaded_file)
    img_w, img_h = pil_image.size
    # 固定一个较小的显示宽度，方便对照
    display_width = 500  
    scale_factor = display_width / img_w
    display_height = int(img_h * scale_factor)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ 对照参考图 (Reference)")
        # 强制显示一张原图作为参考
        st.image(pil_image, width=display_width)
        
        st.divider()
        
        st.subheader("2️⃣ 在此画布上框选样本 (Draw Here)")
        # 画布组件
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_color="#00FF00",
            stroke_width=2,
            background_image=pil_image, # 期望这里能正常显示背景
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="canvas_ai_teaching",
        )

    with col2:
        st.subheader("3️⃣ AI 识别结果 (Result)")
        result_container = st.empty()

        if run_button:
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                with st.spinner("Gemini 正在学习样本并搜索..."):
                    objects = canvas_result.json_data["objects"]
                    user_boxes = []
                    for obj in objects:
                        user_boxes.append({
                            'left': obj['left'] / scale_factor,
                            'top': obj['top'] / scale_factor,
                            'width': obj['width'] / scale_factor,
                            'height': obj['height'] / scale_factor
                        })

                    result_image, count, error = db.detect_similar_buds(st.session_state.api_key, pil_image, user_boxes)
                    
                    if error:
                        st.error(f"发生错误: {error}")
                    else:
                        result_container.image(result_image, use_column_width=True)
                        st.success(f"✅ 分析完成！找到 {count} 个相似目标。")
            else:
                st.warning("⚠️ 请先在左侧画布上框选一个样本。")

elif not st.session_state.api_key:
     st.info("👈 请输入 API Key。")
else:
    st.info("👈 请上传图片。")
