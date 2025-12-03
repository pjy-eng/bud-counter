import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import json
import re

# 使用性能更强的 2.0 Flash 模型
MODEL_NAME = 'gemini-2.0-flash'

def parse_json_from_markdown(text):
    """从 Markdown 代码块中提取 JSON"""
    try:
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception as e:
        print(f"JSON 解析失败: {e}")
        return []

def crop_image(pil_image, box):
    """根据边界框裁剪图片"""
    # box 格式: {'left': x, 'top': y, 'width': w, 'height': h}
    x, y, w, h = int(box['left']), int(box['top']), int(box['width']), int(box['height'])
    # 增加一点点 padding，让样本包含一点周围环境
    padding = 2
    return pil_image.crop((x - padding, y - padding, x + w + padding, y + h + padding))

def detect_similar_buds(api_key, pil_image, user_boxes):
    """
    将用户圈选的样本和原图一起发送给 Gemini，进行相似目标检测。
    """
    if not user_boxes:
        return np.array(pil_image), 0, "请至少圈选一个样本。"

    # 1. 配置 API
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        return None, 0, f"API 配置失败: {e}"

    # 2. 准备输入数据
    # 裁剪出用户圈选的第一个样本作为参考 (我们只取第一个，通常足够了)
    sample_image = crop_image(pil_image, user_boxes[0])
    
    # 构造多模态 Prompt：文本 + 原图 + 样本图
    prompt_parts = [
        "Look at the following two images.",
        "The first image is a microscope image of a cell.",
        "The second image is a small sample crop showing exactly what a 'bud' looks like.",
        pil_image,    # 原图
        sample_image, # 样本图
        """
        Your task is to find ALL instances in the first image that look similar in structure, texture, and shape to the 'bud' shown in the sample image.
        Ignore the main body of the cell, find only the budding structures.
        
        Return a JSON object with a key "boxes". 
        The value should be a list of bounding boxes for each bud found.
        Each bounding box must be in the format [ymin, xmin, ymax, xmax], 
        where the coordinates are normalized to a scale of 0 to 1000 (0 is top/left, 1000 is bottom/right).
        """
    ]

    try:
        # 3. 发送请求
        response = model.generate_content(prompt_parts)
        result_text = response.text
        print("Gemini Raw Response:", result_text)

        # 4. 解析和绘图
        data = parse_json_from_markdown(result_text)
        boxes = data.get("boxes", [])
        
        cv_image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        h_img, w_img, _ = cv_image.shape
        
        count = 0
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            left = int((xmin / 1000) * w_img)
            top = int((ymin / 1000) * h_img)
            right = int((xmax / 1000) * w_img)
            bottom = int((ymax / 1000) * h_img)
            
            # 画框 (绿色, 线宽 2)
            cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 255, 0), 2)
            count += 1
            
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), count, None

    except Exception as e:
        return np.array(pil_image), 0, str(e)
