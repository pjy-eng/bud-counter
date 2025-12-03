import cv2
import numpy as np
from PIL import Image

def find_similar_buds(pil_image, user_boxes, threshold=0.7):
    """
    根据用户圈选的样本框，在整张图中寻找相似的区域。

    Args:
        pil_image: 用户上传的原始 PIL 图片。
        user_boxes: 一个列表，包含用户画的框的字典 [{'left': x, 'top': y, 'width': w, 'height': h}, ...]。
        threshold: 匹配相似度阈值 (0-1)，越高越严格。

    Returns:
        processed_image: 绘制了所有检测结果的 OpenCV 图像。
        count: 检测到的总数（包括用户圈选的）。
    """
    # 1. 图像预处理
    # 转换为 OpenCV 格式 (BGR) 用于处理
    cv_image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    processed_image = cv_image.copy()
    # 转换为灰度图用于模板匹配
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    detected_boxes = []

    # 2. 遍历用户圈选的每一个样本框
    for box in user_boxes:
        # 提取边界框坐标 (确保在图片范围内)
        x = max(0, int(box['left']))
        y = max(0, int(box['top']))
        w = int(box['width'])
        h = int(box['height'])
        
        # 忽略无效的框
        if w <= 0 or h <= 0: continue

        # 从灰度图中裁剪出模板区域
        template = gray_image[y:y+h, x:x+w]
        template_h, template_w = template.shape[:2]
        
        # 3. 执行模板匹配
        # 使用标准相关系数匹配法 (TM_CCOEFF_NORMED)，结果在 0-1 之间
        try:
            res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        except Exception as e:
            print(f"模板太小或在边界外，跳过: {e}")
            continue

        # 4. 筛选高匹配度区域
        # 找到大于设定阈值的所有位置
        loc = np.where(res >= threshold)
        
        # 将匹配点转换为边界框格式 [x, y, w, h]
        for pt in zip(*loc[::-1]): # loc 是 (y, x)，需要反转为 (x, y)
            detected_boxes.append([int(pt[0]), int(pt[1]), template_w, template_h])

    # 5. 去除重叠框 (非极大值抑制 - NMS)
    # 这是一个关键步骤，防止对同一个目标画出多个框
    if not detected_boxes:
        return cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), 0

    detected_boxes = np.array(detected_boxes)
    boxes_for_nms = detected_boxes[:, :4].tolist() # 只取 x, y, w, h
    scores = np.ones(len(boxes_for_nms)).tolist() # 这里暂时假设所有框的得分一样
    
    # NMS 阈值，重叠度大于 0.3 的框会被合并
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, score_threshold=threshold, nms_threshold=0.3)

    final_count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes_for_nms[i]
            # 在结果图上画矩形框 (红色，线宽 2)
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            final_count += 1

    # 将最终结果转换回 RGB 格式供 Streamlit 显示
    return cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), final_count
