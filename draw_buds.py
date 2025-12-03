import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import json
import re

def parse_json_from_markdown(text):
    """从 Markdown 代码块中提取 JSON"""
    try:
        # 尝试通过正则寻找 ```json ... ```
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # 如果没有代码块，尝试直接解析
        return json.loads(text)
    except Exception as e:
        print(f"JSON 解析失败: {e}")
        return []

def detect_with_gemini(api_key, pil_image):
    """
    调用 Gemini 1.5 Flash 识别细胞出芽，并返回绘制好框的图片。
    """
    # 1. 配置 API
    genai.configure(api_key=api_key)
    
    # 使用 Flash 模型，速度快且便宜（对于视觉任务通常足够）
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 2. 构造 Prompt (提示词)
    # 关键在于要求它返回 normalized bounding boxes (0-1000)
    prompt = """
    Look at this microscope image of yeast cells. 
    Identify the "buds" (daughter cells growing off the larger mother cells).
    Ignore the large mother cells, only find the buds.
    
    Return a JSON object with a key "boxes". 
    The value should be a list of bounding boxes for each bud found.
    Each bounding box must be in the format [ymin, xmin, ymax, xmax], 
    where the coordinates are normalized to a scale of 0 to 1000 (0 is top/left, 1000 is bottom/right).
    
    Example output format:
    ```json
    {
      "boxes": [
         [100, 200, 150, 250],
         [500, 600, 550, 650]
      ]
    }
    ```
    """

    try:
        # 3. 发送请求
        response = model.generate_content([prompt, pil_image])
        result_text = response.text
        print("Gemini Raw Response:", result_text) # 调试用

        # 4. 解析坐标
        data = parse_json_from_markdown(result_text)
        boxes = data.get("boxes", [])
        
        # 5. 在原图上绘图
        # 转换图片格式 PIL -> OpenCV
        cv_image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        h_img, w_img, _ = cv_image.shape
        
        count = 0
        for box in boxes:
            # Gemini 返回的是 [ymin, xmin, ymax, xmax] (0-1000 scale)
            ymin, xmin, ymax, xmax = box
            
            # 转换为实际像素坐标
            left = int((xmin / 1000) * w_img)
            top = int((ymin / 1000) * h_img)
            right = int((xmax / 1000) * w_img)
            bottom = int((ymax / 1000) * h_img)
            
            # 画框 (绿色, 线宽 2)
            cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 255, 0), 2)
            # 标号
            cv2.putText(cv_image, f"Bud {count+1}", (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count += 1
            
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), count, None

    except Exception as e:
        return np.array(pil_image), 0, str(e)
