import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Few-Shot Bud Counter", layout="wide")

if 'positive_points' not in st.session_state:
    st.session_state['positive_points'] = []

# ==========================================
# 核心算法：基于多点特征的 One-Class SVM 学习
# ==========================================
def extract_features_around_point(img_gray, x, y, window_size=20):
    """
    在点击点周围提取特征：
    1. 局部平均灰度
    2. 局部方差 (纹理复杂度)
    3. 局部梯度 (边缘强度)
    """
    h, w = img_gray.shape
    x, y = int(x), int(y)
    
    # 边界保护
    y1 = max(0, y - window_size)
    y2 = min(h, y + window_size)
    x1 = max(0, x - window_size)
    x2 = min(w, x + window_size)
    
    patch = img_gray[y1:y2, x1:x2]
    
    if patch.size == 0: return np.zeros(3)
    
    mean_val = np.mean(patch)
    std_val = np.std(patch)
    
    # 简单梯度
    sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    return np.array([mean_val, std_val, grad_mag])

def train_and_predict(img_gray, points, params):
    # 1. 准备训练数据
    features = []
    for p in points:
        feat = extract_features_around_point(img_gray, p[0], p[1], params['window_size'])
        features.append(feat)
    
    X_train = np.array(features)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 2. 训练 One-Class SVM (只学习"什么是Bud")
    # nu 参数控制异常值的比例，gamma 控制核函数的范围
    clf = OneClassSVM(kernel='rbf', nu=params['nu'], gamma=params['gamma'])
    clf.fit(X_train_scaled)
    
    # 3. 全图滑动窗口预测 (为了速度，步长设大一点)
    step = params['step']
    win = params['window_size']
    h, w = img_gray.shape
    
    found_points = []
    
    # 这里的循环如果用 Python 写会很慢，但为了演示逻辑先这样
    # 实际部署时，这里只会在关键点附近采样，或者使用图像处理方法加速
    # 改进策略：先用简单的阈值筛选出候选点，再用 SVM 确认
    
    # 快速预筛选：基于训练样本的平均亮度
    mean_intensity = np.mean(X_train[:, 0])
    lower_bound = mean_intensity - 30
    upper_bound = mean_intensity + 30
    
    # 二值化找到大概区域
    _, mask = cv2.threshold(img_gray, lower_bound, 255, cv2.THRESH_BINARY)
    # 结合方差（纹理）筛选
    # 这里简化为：只在 mask 为白色的区域采样
    
    y_indices, x_indices = np.where(mask > 0)
    
    # 随机采样或者间隔采样以提高速度
    # 我们改用简单的滑动窗口策略，但只在可能有东西的地方滑
    
    # 为了演示实时性，我们退回到更简单的 "多模板匹配逻辑"
    # SVM 在纯 Python 循环里太慢了。
    # 方案 B：多点平均模板匹配
    
    return adaptive_multi_template_matching(img_gray, points, params)


def adaptive_multi_template_matching(img_gray, points, params):
    """
    替代 SVM 的快速方案：
    在每个点击位置截取一个小模板，算出平均模板，然后全图搜。
    """
    win = params['window_size']
    h, w = img_gray.shape
    templates = []
    
    # 1. 收集所有点击处的模板
    for p in points:
        x, y = int(p[0]), int(p[1])
        y1, y2 = max(0, y-win), min(h, y+win)
        x1, x2 = max(0, x-win), min(w, x+win)
        patch = img_gray[y1:y2, x1:x2]
        if patch.shape == (2*win, 2*win): # 确保尺寸一致
            templates.append(patch)
            
    if not templates: return [], img_gray
    
    # 2. 计算平均模板 (这是关键！融合了多个样本的特征)
    avg_template = np.mean(templates, axis=0).astype(np.uint8)
    
    # 3. 匹配
    res = cv2.matchTemplate(img_gray, avg_template, cv2.TM_CCOEFF_NORMED)
    
    # 4. 筛选结果
    loc = np.where(res >= params['threshold'])
    boxes = []
    w_t, h_t = avg_template.shape[::-1]
    
    for pt in zip(*loc[::-1]):
        boxes.append([int(pt[0]), int(pt[1]), w_t, h_t])
        
    rects, _ = cv2.groupRectangles(boxes, groupThreshold=1, eps=0.3)
    
    res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_buds = []
    
    for (x, y, w_box, h_box) in rects:
        # 绘制
        cv2.rectangle(res_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
        final_buds.append([x, y])
        
    # 标记用户点击的点
    for p in points:
        cv2.circle(res_img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        
    return final_buds, res_img


# ==========================================
# UI 布局
# ==========================================
st.sidebar.header("🎛️ 参数设置")
win_size = st.sidebar.slider("样本半径 (Window Size)", 10, 50, 20, help="点击点周围多大范围内算作一个样本")
thresh = st.sidebar.slider("相似度阈值", 0.3, 0.95, 0.60)

st.title("👆 点选学习版 (Point & Find)")
st.markdown("思路：**不要画框，直接点击** 3-5 个典型的 Bud，系统会计算它们的**平均特征**去找剩下的。")

uploaded_file = st.file_uploader("上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1. 点击样本 (Point)")
        st.caption("请用鼠标左键点击图中的 Bud 中心。点 3 个以上效果最好。")
        
        # Point 模式
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 1)",
            stroke_color="#00FF00",
            background_image=pil_img,
            update_streamlit=True,
            height=500,
            drawing_mode="point", # 关键：点选模式
            point_display_radius=5,
            key="canvas_point"
        )

    with col2:
        st.subheader("2. 学习与搜索")
        
        # 获取点击点
        if canvas.json_data and len(canvas.json_data["objects"]) > 0:
            points = []
            for obj in canvas.json_data["objects"]:
                points.append([obj['left'], obj['top']])
            
            st.info(f"已采集 {len(points)} 个样本点")
            
            if len(points) >= 1:
                params = {'window_size': win_size, 'threshold': thresh}
                
                # 运行多点匹配
                buds, res_img = adaptive_multi_template_matching(img_gray, points, params)
                
                st.metric("✅ 找到相似目标", f"{len(buds)} 个")
                st.image(res_img, use_column_width=True, caption="绿点=你的样本，红框=AI找到的")
            else:
                st.warning("请至少点击 1 个点。")
        else:
            st.info("👈 请在左图点击 Bud 中心。")
