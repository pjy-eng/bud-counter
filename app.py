import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="Immersive Bud Counter", layout="wide")

# ==========================================
# 1. 核心算法：经典模板匹配 (复刻 Image 2)
# ==========================================
def run_template_matching(img_gray, roi_coords, threshold):
    # 1. 预处理 (CLAHE 增强)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    
    # 2. 提取模板
    rx, ry, rw, rh = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
    
    h, w = img_enhanced.shape
    if rw <= 5 or rh <= 5 or rx < 0 or ry < 0:
        return [], img_enhanced, "框选太小，请重画。"

    template = img_enhanced[ry:ry+rh, rx:rx+rw]
    
    # 3. 核心匹配 (TM_CCOEFF_NORMED)
    res = cv2.matchTemplate(img_enhanced, template, cv2.TM_CCOEFF_NORMED)
    
    # 4. 筛选与去重
    loc = np.where(res >= threshold)
    boxes = []
    for pt in zip(*loc[::-1]):
        boxes.append([int(pt[0]), int(pt[1]), rw, rh])
        
    # NMS 去重
    rects, _ = cv2.groupRectangles(boxes, groupThreshold=1, eps=0.3)
    
    # 5. 绘图
    res_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_buds = []
    
    user_center = (rx + rw//2, ry + rh//2)
    
    for (x, y, w_box, h_box) in rects:
        # 排除用户自己画的那个框
        curr_center = (x + w_box//2, y + h_box//2)
        dist = np.sqrt((user_center[0]-curr_center[0])**2 + (user_center[1]-curr_center[1])**2)
        
        if dist < rw / 2:
            continue
            
        final_buds.append([x, y])
        # 画红框
        cv2.rectangle(res_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
        
    # 画绿框
    cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
    
    return final_buds, res_img, ""

# ==========================================
# 2. UI 布局 (完美修复版)
# ==========================================
st.title("🔬 沉浸式 Bud 计数器")
st.caption("现在，图片就是画布。请直接在左图上 **画框**。")

# 侧边栏
st.sidebar.header("🎛️ 参数")
threshold = st.sidebar.slider("相似度阈值", 0.3, 0.95, 0.60, help="如果漏检，调低；如果误检，调高。")

uploaded_file = st.file_uploader("上传图像", type=["jpg", "png", "tif"])

if uploaded_file:
    # 1. 加载并计算尺寸
    pil_img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = pil_img.size
    
    # === 关键修改：计算适应屏幕的显示尺寸 ===
    # 我们将宽度固定为 700px (这是一个在网页上看起来比较舒服的宽度)
    # 然后按比例计算高度
    display_width = 700
    ratio = display_width / orig_w
    display_height = int(orig_h * ratio)
    
    # 缩放图片，用于显示在 Canvas 上
    # 注意：后续的算法处理也必须基于这张缩放后的图，否则坐标会对不上
    pil_img_resized = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
    img_array = np.array(pil_img_resized)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("1. 框选模板")
        st.caption("请画框包围一个标准的 Bud。")
        
        # === 关键修改：画布尺寸与图片显示尺寸完全一致 ===
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_color="#00FF00",
            background_image=pil_img_resized, # 背景图铺满
            update_streamlit=True,
            width=display_width,   # 强制宽度无缝贴合
            height=display_height, # 强制高度无缝贴合
            drawing_mode="rect",   # 回归画框模式
            key="canvas_final"
        )

    with col2:
        st.subheader("2. 结果")
        
        # 获取画框数据
        if canvas.json_data and len(canvas.json_data["objects"]) > 0:
            obj = canvas.json_data["objects"][-1]
            roi = {
                'left': int(obj['left']), 
                'top': int(obj['top']), 
                'width': int(obj['width']), 
                'height': int(obj['height'])
            }
            
            # 只有当框有效时才计算
            if roi['width'] > 5 and roi['height'] > 5:
                with st.spinner("分析中..."):
                    buds, res_img, msg = run_template_matching(img_gray, roi, threshold)
                
                # 计数逻辑：找到的 + 模板自己
                total = len(buds) + 1
                st.metric("✅ 总计数", f"{total} 个")
                
                # 显示结果
                st.image(res_img, use_column_width=True, caption="绿框=模板，红框=找到的")
            else:
                st.warning("框太小了，请重画。")
        else:
            st.info("👈 请在左图直接画框。")

else:
    st.info("请上传图片。")
