import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def base64_to_cv2(image_data):
    """将前端传来的 base64 图片转换为 OpenCV 格式"""
    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_base64(img):
    """将 OpenCV 图片转换为 base64 返回前端"""
    _, buffer = cv2.imencode('.jpg', img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')


def calculate_circularity(area, perimeter):
    if perimeter == 0: return 0
    return (4 * np.pi * area) / (perimeter ** 2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    data = request.json
    image_data = data['image']
    roi = data['roi']  # {x, y, w, h}

    # 1. 读取图像
    img_bgr = base64_to_cv2(image_data)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 预处理：CLAHE 增强对比度 (这对显微图像至关重要)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    img_blur = cv2.GaussianBlur(img_enhanced, (5, 5), 0)

    # 3. 分析用户选定的 ROI (模板)
    rx, ry, rw, rh = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])

    # 边界保护
    if rw <= 0 or rh <= 0: return jsonify({'error': 'ROI invalid'})

    roi_region = img_blur[ry:ry + rh, rx:rx + rw]

    # 在 ROI 内部进行 Otsu 阈值分割，找出前景
    _, roi_thresh = cv2.threshold(roi_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 寻找 ROI 里最大的轮廓作为“标准芽”
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({'message': 'ROI中未检测到明显物体', 'count': 0})

    template_cnt = max(contours, key=cv2.contourArea)

    # 提取模板特征
    tmpl_area = cv2.contourArea(template_cnt)
    tmpl_perimeter = cv2.arcLength(template_cnt, True)
    tmpl_circularity = calculate_circularity(tmpl_area, tmpl_perimeter)
    # 计算 ROI 区域的平均亮度
    tmpl_mean_val = cv2.mean(roi_region, mask=roi_thresh)[0]

    # 4. 全图分割与匹配
    # 对全图应用二值化 (使用 ROI 计算出的 Otsu 阈值或自适应阈值)
    # 这里为了鲁棒性，使用自适应阈值
    thresh_global = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2
    )

    # 形态学操作：开运算（断开粘连）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_global = cv2.morphologyEx(thresh_global, cv2.MORPH_OPEN, kernel, iterations=2)

    # 寻找所有轮廓
    all_contours, _ = cv2.findContours(thresh_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matched_buds = []

    # 5. 筛选逻辑 (Feature Matching)
    # 容差设置 (可以做成前端滑块)
    area_tol = 0.45  # 面积容差 ±45%
    circ_tol = 0.30  # 圆度容差 ±30% (应对形变)

    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = calculate_circularity(area, perimeter)

        # 1. 面积筛选
        if not (tmpl_area * (1 - area_tol) < area < tmpl_area * (1 + area_tol)):
            continue

        # 2. 圆度筛选 (排除细长的杂质)
        if not (tmpl_circularity * (1 - circ_tol) < circularity):
            continue

        matched_buds.append(cnt)

    # 6. 绘制结果
    result_img = img_bgr.copy()

    # 画出所有的匹配项 (红色填充，半透明效果很难在后端做，这里画轮廓)
    cv2.drawContours(result_img, matched_buds, -1, (0, 0, 255), 2)

    # 画出用户选的 ROI (绿色矩形)
    cv2.rectangle(result_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    # 标注总数
    count = len(matched_buds)
    cv2.putText(result_img, f"Count: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return jsonify({
        'result_image': cv2_to_base64(result_img),
        'count': count,
        'debug_info': {
            'template_area': tmpl_area,
            'template_circ': tmpl_circularity
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)