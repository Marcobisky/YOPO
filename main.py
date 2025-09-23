import cv2
import numpy as np
import os
from glob import glob

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {path}")
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def compute_color_hist(img, mask=None):
    # 使用 HSV 色彩直方图
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 只取 H 和 S 通道
    h_bins = 30
    s_bins = 32
    hist = cv2.calcHist([hsv], [0,1], mask, [h_bins, s_bins], [0,180, 0,256])
    cv2.normalize(hist, hist)
    return hist

def color_similarity(hist1, hist2):
    # 比较两个直方图，例如用相关性或巴氏距离
    # 这里用相关性（Correlation）
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def normalized_cross_correlation(template_gray, patch_gray):
    # 使用 OpenCV 自带函数
    # cv2.matchTemplate 要求 template 比 patch 小或者等
    res = cv2.matchTemplate(patch_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    # 返回最大值和其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def crop(img, center, size):
    """
    从 img 中裁剪一个子图，中心为 center (x, y)，大小 size (w, h)
    如果超出边界，需要做边缘处理
    """
    x_center, y_center = center
    w, h = size
    x1 = int(x_center - w/2)
    y1 = int(y_center - h/2)
    x2 = x1 + w
    y2 = y1 + h
    # 边界检查
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(img.shape[1], x2)
    y2_clip = min(img.shape[0], y2)
    patch = img[y1_clip:y2_clip, x1_clip:x2_clip]
    return patch, (x1_clip, y1_clip)

def track_sequence(sequence_folder, init_frame_idx, init_bbox, output_info_path=None):
    """
    跟踪一个 sequence
    sequence_folder: e.g. "./sequences/blue"
    init_frame_idx: 第一帧的编号，比如 1085
    init_bbox: 初始边框 (x, y, w, h) 在第一帧中选定
    output_info_path: 可选，将每帧 bbox 和得分等写日志
    """
    # 参数设定
    ncc_threshold = 0.7
    color_threshold = 0.5
    alpha = 0.2  # 模板更新率
    window_expand = 50  # 搜索窗口在预测中心周围展开像素范围
    step_size = 5       # 搜索窗口内滑动步长（粗调）

    # 准备文件列表
    img_paths = sorted(glob(os.path.join(sequence_folder, "cars_*.jpg")))
    # 根据 init_frame_idx 筛出
    img_paths = [p for p in img_paths if int(os.path.basename(p).split('_')[1].split('.')[0]) >= init_frame_idx]

    # 第1帧初始化
    first_path = img_paths[0]
    first_img = load_image(first_path)
    template_color = first_img[init_bbox[1]:init_bbox[1]+init_bbox[3], init_bbox[0]:init_bbox[0]+init_bbox[2]]
    template_gray = to_gray(template_color)
    template_hist = compute_color_hist(template_color)

    # 状态变量
    pos_prev = (init_bbox[0] + init_bbox[2]/2, init_bbox[1] + init_bbox[3]/2)  # 中心点
    pos_prev2 = pos_prev
    velocity = (0.0, 0.0)
    template_size = (init_bbox[2], init_bbox[3])

    log = []  # 用来保存每帧结果

    for idx, img_path in enumerate(img_paths[1:], start=2):
        frame = load_image(img_path)
        frame_gray = to_gray(frame)

        # 1. 运动预测
        # 这里用匀速模型：pos_pred = pos_prev + velocity
        pos_pred = (pos_prev[0] + velocity[0], pos_prev[1] + velocity[1])

        # 2. 搜索窗口
        w, h = template_size
        x_center_pred, y_center_pred = pos_pred
        x1_sw = int(x_center_pred - w/2 - window_expand)
        y1_sw = int(y_center_pred - h/2 - window_expand)
        x2_sw = int(x_center_pred + w/2 + window_expand)
        y2_sw = int(y_center_pred + h/2 + window_expand)
        # 边界控制
        x1_sw = max(0, x1_sw)
        y1_sw = max(0, y1_sw)
        x2_sw = min(frame.shape[1] - 1, x2_sw)
        y2_sw = min(frame.shape[0] - 1, y2_sw)

        best_score = -1.0
        best_center = pos_prev  # 如果失败，维持上一帧
        best_bbox = None

        # 3‑4. 搜索窗口内滑动候选 + 可选颜色筛选
        for x in range(x1_sw, x2_sw - w + 1, step_size):
            for y in range(y1_sw, y2_sw - h + 1, step_size):
                patch_color = frame[y:y+h, x:x+w]
                if patch_color.shape[0] != h or patch_color.shape[1] != w:
                    continue

                # 颜色辅助筛选
                patch_hist = compute_color_hist(patch_color)
                c_sim = color_similarity(patch_hist, template_hist)
                if c_sim < color_threshold:
                    continue

                # NCC 匹配
                patch_gray = frame_gray[y:y+h, x:x+w]
                score, _ = normalized_cross_correlation(template_gray, patch_gray)
                if score > best_score:
                    best_score = score
                    best_center = (x + w/2, y + h/2)
                    best_bbox = (x, y, w, h)

        # 6. 检查可靠性
        if best_score < ncc_threshold or best_bbox is None:
            # 匹配不可靠
            # 可以选择保持上一帧的 bbox，velocity 不变或稍微衰减
            pos_k = pos_prev
            bbox_k = (int(pos_prev[0] - w/2), int(pos_prev[1] - h/2), w, h)
        else:
            pos_k = best_center
            bbox_k = best_bbox

        # 7. 更新速度与位置
        velocity = (pos_k[0] - pos_prev[0], pos_k[1] - pos_prev[1])
        pos_prev2 = pos_prev
        pos_prev = pos_k

        # 8. 模板更新
        if best_score >= ncc_threshold + 0.1:  # 比阈值还要高一点才更新
            # 获取当前匹配 patch
            matched_patch_color = frame[int(bbox_k[1]):int(bbox_k[1]+h), int(bbox_k[0]):int(bbox_k[0]+w)]
            matched_patch_gray = frame_gray[int(bbox_k[1]):int(bbox_k[1]+h), int(bbox_k[0]):int(bbox_k[0]+w)]
            # 更新模板灰度
            template_gray = cv2.addWeighted(template_gray.astype(np.float32), 1 - alpha,
                                            matched_patch_gray.astype(np.float32), alpha, 0).astype(np.uint8)
            # 更新颜色直方图
            hist_new = compute_color_hist(matched_patch_color)
            # 简单滑动平均
            template_hist = (1 - alpha) * template_hist + alpha * hist_new

        # 9. 输出结果：记录 log + 可视化
        log.append({
            "frame_path": img_path,
            "bbox": bbox_k,
            "score": best_score
        })

        # 可选可视化
        # 在 frame 上画 bbox_k
        x0, y0, ww, hh = bbox_k
        cv2.rectangle(frame, (x0, y0), (x0 + ww, y0 + hh), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_score:.2f}", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # 写 log 到文件
    if output_info_path is not None:
        with open(output_info_path, 'w') as f:
            for x in log:
                f.write(f"{os.path.basename(x['frame_path'])}, {x['bbox'][0]}, {x['bbox'][1]}, {x['bbox'][2]}, {x['bbox'][3]}, {x['score']:.4f}\n")
    return log

if __name__ == "__main__":
    # 示例：追踪 blue sequence
    # 在第 1 帧你需要人工给初始 bbox
    # 比如假设你选定 blue 的第 1 帧 cars_1085.jpg, bbox 手动选 (x, y, w, h)
    blue_init_frame = 1085
    blue_bbox = (677, 257, 163, 132)  # 示例，需要你自己调整为实际目标 bbox

    red_init_frame = 1517
    red_bbox = (791, 269, 195, 151)   # 示例，需要你自己选

    # 跟踪 blue
    log_blue = track_sequence("./sequences/blue", blue_init_frame, blue_bbox, output_info_path="blue_tracking_log.txt")
    # 跟踪 red
    log_red = track_sequence("./sequences/red", red_init_frame, red_bbox, output_info_path="red_tracking_log.txt")