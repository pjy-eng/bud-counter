import google.generativeai as genai
import os

# 替换为你的真实 API Key，或者设置环境变量
api_key = "AIzaSyCfIzL3xFXqlNuQCY_0EiAn-E7qu7yuv0g"

genai.configure(api_key=api_key)

print("正在查询可用模型列表...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"查询失败: {e}")