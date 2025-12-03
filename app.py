import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="High-Accuracy Bud Counter", layout="wide")

if 'roi_coords' not in st.session_state:
    st.session_state['roi_coords'] = None

# ==========================================
# 1. 核心算法：多尺度 + 多角度 模板匹配
# ==========================================
def rotate_image(image, angle):
    """辅助函数：旋转图像"""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def process_multiscale_matching(img_gray, roi_coords, params):
    try:
        # --- A. 预处理 ---
        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8)

        # 对比度增强 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_gray)
        
        # --- B. 准备模板 ---
        rx, ry, rw, rh = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
        h, w = img_enhanced.shape
        
        # 边界检查
        if rw <= 5 or rh <= 5 or rx >= w or ry >= h:
            return None, None, "框选区域无效或太小。"
            
        base_template = img_enhanced[ry:ry+rh, rx:rx+rw]
        
        # --- C. 多尺度 + 多角度搜索 ---
        all_detections = [] # 存储格式: [x, y, w, h, score]
        
        # 1. 定义搜索范围
        # 尺度：从 0.8 倍到 1.2 倍，分 5 档
        scales = np.linspace(0.8, 1.2, 5) 
        # 角度：0, 90, 180, 270 (如果需要更精细可以加 45, 135...)
        angles = [0, 90, 180, 270] if params['use_rotation'] else [0]
        
        threshold = params['match_thresh']

        # 2. 循环匹配 (暴力搜索)
        for scale in scales:
            # 缩放模板
            t_w = int(base_template.shape[1] * scale)
            t_h = int(base_template.shape[0] * scale)
            
            if t_w <= 0 or t_h <= 0 or t_w > w or t_h > h: continue
            
            scaled_template_base = cv2.resize(base_template, (t_w, t_h))
            
            for angle in angles:
                # 旋转模板
                if angle == 0:
                    curr_template = scaled_template_base
                else:
                    curr_template = rotate_image(scaled_template_base, angle)

                # 匹配
                res = cv2.matchTemplate(img_enhanced, curr_template, cv2.TM_CCOEFF_NORMED)
                
                # 筛选
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    # 记录结果：x, y, w, h, score
                    score = res[pt[1], pt[0]]
                    all_detections.append([int(pt[0]), int(pt[1]), t_w, t_h, score])

        # --- D. NMS (非极大值抑制) 去重 ---
        if not all_detections:
            return [], cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), "未找到匹配目标，请降低阈值。"

        # 将 list 转为 numpy array 以便处理
        detections = np.array(all_detections)
        
        # OpenCV 的 groupRectangles 需要 [x, y, w, h] 格式
        # 但我们需要保留 score 来做更高级的筛选，这里手写一个简单的基于 score 的 NMS
        # 或者为了简单稳定，使用 OpenCV 的 groupRectangles (不考虑 score，只考虑位置)
        
        # 转换格式适配 cv2.groupRectangles
        rects_for_cv = []
        for det in detections:
            rects_for_cv.append([int(det[0]), int(det[1]), int(det[2]), int(det[3])])
        
        # groupThreshold=1: 至少重叠 1 次 (去噪)
        # eps=0.2: 重叠阈值
        nms_rects, weights = cv2.groupRectangles(rects_for_cv, groupThreshold=1, eps=0.2)
        
        # --- E. 绘图与排除自身 ---
        res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        final_buds = []
        
        user_center = (rx + rw//2, ry + rh//2)
        
        for (x, y, w_box, h_box) in nms_rects:
            # 计算中心距离，排除用户自己画的那个框
            curr_center = (x + w_box//2, y + h_box//2)
            dist = np.sqrt((user_center[0]-curr_center[0])**2 + (user_center[1]-curr_center[1])**2)
            
            if dist < rw / 2: # 如果非常接近原点，跳过
                continue
                
            final_buds.append([x, y, w_box, h_box])
            
            # 画红框 (显示找到的目标)
            cv2.rectangle(res_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

        # 画绿框 (用户模板)
        cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        
        return final_buds, res_img, f"搜索完成"

    except Exception as e:
        return None, None, f"算法错误: {str(e)}"

# ==========================================
# 2. UI 布局
# ==========================================
st.sidebar.header("🎛️ 高级设置")

params = {
    'match_thresh': st.sidebar.slider("相似度阈值", 0.3, 0.95, 0.60, 0.01, help="越低越容易找到（可能误报），越高越精准"),
    'use_rotation': st.sidebar.checkbox("启用旋转搜索 (更准但更慢)", value=False, help="勾选后会尝试不同角度匹配，耗时增加 4 倍")
}

st.title("🔬 高精度细胞计数 (多尺度版)")
st.markdown("此版本会自动搜索 **大小不同 (±20%)** 的目标。勾选左侧 **旋转搜索** 可进一步提高准确率。")

uploaded_file = st.file_uploader("1. 上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("2. 定义模板")
        st.caption("框选一个清晰的 Bud 作为基准。")
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_color="#00FF00",
            background_image=pil_img,
            update_streamlit=True,
            height=500,
            drawing_mode="rect",
            key="canvas_multi",
        )

    with col2:
        st.subheader("3. 智能分析")
        
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            roi_coords = {
                'left': int(obj['left']), 'top': int(obj['top']),
                'width': int(obj['width']), 'height': int(obj['height'])
            }
            
            if roi_coords['width'] > 0:
                with st.spinner("正在进行多尺度全图扫描..."):
                    buds, res_img, msg = process_multiscale_matching(img_gray, roi_coords, params)

                if buds is not None:
                    count = len(buds) + 1
                    st.metric("✅ 最终计数", f"{count} 个")
                    
                    fig = px.imshow(res_img)
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"已自动匹配 0.8x ~ 1.2x 大小的目标")
                else:
                    st.warning(msg)
        else:
            st.info("👈 请先画框。")

else:
    st.info("请上传图片。")
