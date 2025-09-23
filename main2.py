import cv2
import numpy as np
import os
from glob import glob

from main import track_sequence

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {path}")
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def compute_color_hist(img, mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def color_similarity(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def normalized_cross_correlation(template, patch):
    res = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def resize_template(template, size):
    return cv2.resize(template, size, interpolation=cv2.INTER_LINEAR)

def track_sequence_with_scale(sequence_folder, init_frame_idx, init_bbox, output_info_path=None):
    img_paths = sorted(glob(os.path.join(sequence_folder, "cars_*.jpg")))
    img_paths = [p for p in img_paths if int(os.path.basename(p).split('_')[1].split('.')[0]) >= init_frame_idx]

    first_path = img_paths[0]
    first_img = load_image(first_path)
    x, y, w0, h0 = init_bbox
    template_color = first_img[y:y+h0, x:x+w0]
    template_gray_orig = to_gray(template_color)
    template_hist = compute_color_hist(template_color)

    pos_prev = (x + w0 / 2, y + h0 / 2)
    pos_prev2 = pos_prev
    velocity = (0.0, 0.0)
    scale_prev = 1.0
    scale_prev2 = 1.0
    scale_velocity = 0.0
    alpha = 0.2
    window_expand = 50
    step_size = 5
    scale_candidates = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
    ncc_threshold = 0.7
    color_threshold = 0.5

    log = []

    for idx, img_path in enumerate(img_paths[1:], start=2):
        frame = load_image(img_path)
        frame_gray = to_gray(frame)

        scale_pred = scale_prev + scale_velocity
        pos_pred = (pos_prev[0] + velocity[0], pos_prev[1] + velocity[1])

        best_score = -1.0
        best_bbox = None
        best_scale = scale_prev

        for scale in scale_candidates:
            scale_actual = scale_pred * scale
            w = int(w0 * scale_actual)
            h = int(h0 * scale_actual)
            if w < 10 or h < 10 or w > frame.shape[1] or h > frame.shape[0]:
                continue

            x1_sw = int(pos_pred[0] - w/2 - window_expand)
            y1_sw = int(pos_pred[1] - h/2 - window_expand)
            x2_sw = int(pos_pred[0] + w/2 + window_expand)
            y2_sw = int(pos_pred[1] + h/2 + window_expand)

            x1_sw = max(0, x1_sw)
            y1_sw = max(0, y1_sw)
            x2_sw = min(frame.shape[1] - 1, x2_sw)
            y2_sw = min(frame.shape[0] - 1, y2_sw)

            resized_template = resize_template(template_gray_orig, (w, h))

            for x in range(x1_sw, x2_sw - w + 1, step_size):
                for y in range(y1_sw, y2_sw - h + 1, step_size):
                    patch_color = frame[y:y+h, x:x+w]
                    if patch_color.shape[0] != h or patch_color.shape[1] != w:
                        continue
                    hist = compute_color_hist(patch_color)
                    c_sim = color_similarity(hist, template_hist)
                    if c_sim < color_threshold:
                        continue
                    patch_gray = frame_gray[y:y+h, x:x+w]
                    score, _ = normalized_cross_correlation(resized_template, patch_gray)
                    if score > best_score:
                        best_score = score
                        best_bbox = (x, y, w, h)
                        best_scale = scale_actual

        if best_score < ncc_threshold or best_bbox is None:
            pos_k = pos_prev
            scale_k = scale_prev
            bbox_k = (int(pos_prev[0] - w0*scale_prev/2), int(pos_prev[1] - h0*scale_prev/2), int(w0*scale_prev), int(h0*scale_prev))
        else:
            x, y, w, h = best_bbox
            pos_k = (x + w / 2, y + h / 2)
            scale_k = best_scale
            bbox_k = best_bbox

        velocity = (pos_k[0] - pos_prev[0], pos_k[1] - pos_prev[1])
        scale_velocity = scale_k - scale_prev
        pos_prev2 = pos_prev
        pos_prev = pos_k
        scale_prev2 = scale_prev
        scale_prev = scale_k

        if best_score >= ncc_threshold + 0.1:
            matched_patch = frame_gray[bbox_k[1]:bbox_k[1]+bbox_k[3], bbox_k[0]:bbox_k[0]+bbox_k[2]]
            template_gray_orig = cv2.addWeighted(resize_template(template_gray_orig, matched_patch.shape[::-1]).astype(np.float32), 1 - alpha, matched_patch.astype(np.float32), alpha, 0).astype(np.uint8)
            template_hist = (1 - alpha) * template_hist + alpha * compute_color_hist(frame[bbox_k[1]:bbox_k[1]+bbox_k[3], bbox_k[0]:bbox_k[0]+bbox_k[2]])

        log.append({
            "frame_path": img_path,
            "bbox": bbox_k,
            "score": best_score
        })

        frame_vis = frame.copy()
        x0, y0, ww, hh = bbox_k
        cv2.rectangle(frame_vis, (x0, y0), (x0 + ww, y0 + hh), (0, 255, 0), 2)
        cv2.putText(frame_vis, f"{best_score:.2f}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Tracking", frame_vis)
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()

    if output_info_path:
        with open(output_info_path, 'w') as f:
            for x in log:
                name = os.path.basename(x['frame_path'])
                bx, by, bw, bh = x['bbox']
                f.write(f"{name},{bx},{by},{bw},{bh},{x['score']:.4f}\n")

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