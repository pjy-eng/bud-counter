import streamlit as st
import openai
import base64
import json
from utils.draw_buds import draw_buds

st.title("🧠 Bud Detector (ChatGPT Powered)")
st.write("上传 TEM 图像 → 接入 ChatGPT 进行自动 bud 检测")

openai.api_key = st.secrets["OPENAI_API_KEY"]

uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "tif"])

if uploaded:
    st.image(uploaded, caption="输入图像")

    img_bytes = uploaded.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    with open("prompts/bud_prompt.txt", "r") as f:
        prompt = f.read()

    st.write("检测中… 请稍等")

    response = openai.ChatCompletion.create(
        model="gpt-4.1-vision-preview",   # 最新视觉模型
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "请分析此图像"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
            ]}
        ],
        max_tokens=4096,
    )

    result = response.choices[0].message["content"]
    
    # 解析 JSON
    result = json.loads(result)

    st.subheader("Bud Count")
    st.write(result["count"])

    st.subheader("坐标列表")
    st.json(result)

    # 画图
    output_img = draw_buds(img_bytes, result["buds"])
    st.image(output_img, caption="检测结果图")
