import streamlit as st
from PIL import Image
import draw_buds as db

st.set_page_config(page_title="Gemini 细胞出芽识别", page_icon="🧠", layout="wide")

st.title("🧠 Gemini AI 细胞出芽识别系统")
st.markdown("本系统接入了 **Gemini 1.5 Flash** 大模型，能够像人类专家一样通过视觉理解识别细胞出芽。")
st.markdown("---")

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 配置大脑")
    # 安全起见，密码形式输入 Key
    api_key = st.text_input("请输入 Google Gemini API Key:", type="password")
    st.caption("没有 Key? [点击这里申请](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    
    st.header("2. 上传图像")
    uploaded_file = st.file_uploader("选择显微镜图片...", type=['jpg', 'png', 'jpeg'])

# --- 主界面 ---
if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图像")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("AI 识别结果")
        result_placeholder = st.empty()
        
        # 只有当上传了图片且有了 Key 才能运行
        if api_key:
            if st.button("让 Gemini 分析图片 (Start Analysis) ✨", type="primary"):
                with st.spinner("正在连接 Google 大脑进行视觉分析..."):
                    # 调用后端
                    result_img, count, error = db.detect_with_gemini(api_key, image)
                    
                    if error:
                        st.error(f"API 调用出错: {error}")
                    else:
                        result_placeholder.image(result_img, use_column_width=True)
                        st.success(f"分析完成！Gemini 发现了 {count} 个出芽点。")
                        
                        # 简单的解释性输出
                        st.markdown(f"""
                        > **AI 分析报告**: 
                        > 我在图像中识别出了 **{count}** 个正在生长的芽（Daughter Cells）。
                        > 绿色框标出了它们的位置。
                        """)
        else:
            result_placeholder.info("👈 请在左侧侧边栏输入 API Key 以启动 AI。")

else:
    st.info("请上传一张图片开始。")
