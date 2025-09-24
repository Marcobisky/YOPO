import cv2
import numpy as np
import os
from glob import glob

class MatchingEvaluator:
    # Evaluate the matching score between current image and kernel
    def __init__(self, Current_Image, Current_Kernel):
        self.Current_Image = Current_Image
        self.Current_Kernel = Current_Kernel
        
    def compute_color_hist(self, img, mask=None):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_bins, s_bins = 30, 32
        hist = cv2.calcHist([hsv], [0,1], mask, [h_bins, s_bins], [0,180, 0,256])
        cv2.normalize(hist, hist)
        return hist
    
    def color_similarity(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def normalized_cross_correlation(self, template_gray, patch_gray):
        res = cv2.matchTemplate(patch_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc
    
    def evaluate_matching_score(self, patch_pos, kernel_size):
        x, y = patch_pos
        w, h = kernel_size
        patch_color = self.Current_Image.image_content_GRB[y:y+h, x:x+w]
        if patch_color.shape[0] != h or patch_color.shape[1] != w:
            return -1.0
        
        # Color similarity check
        patch_hist = self.compute_color_hist(patch_color)
        kernel_hist = self.compute_color_hist(self.Current_Kernel.image_content_GRB)
        c_sim = self.color_similarity(patch_hist, kernel_hist)
        if c_sim < 0.5:  # color_threshold
            return -1.0
        
        # NCC matching
        patch_gray = self.Current_Image._to_gray()[y:y+h, x:x+w]
        kernel_gray = self.Current_Kernel._to_gray()
        score, _ = self.normalized_cross_correlation(kernel_gray, patch_gray)
        return score

class CurrentImage:
    # The current focused image
    def __init__(self, image_content_GRB):
        self.image_content_GRB = image_content_GRB

    def _to_gray(self):
        return cv2.cvtColor(self.image_content_GRB, cv2.COLOR_BGR2GRAY)
    
    def _get_size(self):
        return self.image_content_GRB.shape[:2]  # (height, width)

class CurrentKernel(CurrentImage):
    # The current kernel image used for pattern matching
    def __init__(self, kernel_content_GRB, pos_and_scale):
        super().__init__(kernel_content_GRB)
        self.pos_and_scale = pos_and_scale  # (x, y, scale)
    
    def get_center_pos(self):
        x, y, scale = self.pos_and_scale
        h, w = self.image_content_GRB.shape[:2]
        return (x + w//2, y + h//2)
    
    def get_size(self):
        return self.image_content_GRB.shape[1], self.image_content_GRB.shape[0]  # (w, h)

class NextKernelExplore:
    # Explore the next kernel entity based on history
    def __init__(self, Current_Image, Kernel_Buffer, length_for_prediction=5):
        self.Current_Image = Current_Image
        self.Kernel_Buffer = Kernel_Buffer
        self.length_for_prediction = length_for_prediction

    def explore_next_kernel(self):
        if len(self.Kernel_Buffer.buffer) == 0:
            return None
        
        current_kernel = self.Kernel_Buffer.buffer[-1]
        w, h = current_kernel.get_size()
        
        # Motion prediction based on velocity
        velocity = self._weighted_velocity_predict()
        prev_center = current_kernel.get_center_pos()
        pred_center = (prev_center[0] + velocity[0], prev_center[1] + velocity[1])
        
        # Search window around predicted position
        window_expand = 40  # 减小搜索窗口，从50减到30
        step_size = 3       # 增大步长，从5增到8，减少搜索点数
        img_h, img_w = self.Current_Image._get_size()
        
        x1_sw = max(0, int(pred_center[0] - w//2 - window_expand))
        y1_sw = max(0, int(pred_center[1] - h//2 - window_expand))
        x2_sw = min(img_w - w, int(pred_center[0] + w//2 + window_expand))
        y2_sw = min(img_h - h, int(pred_center[1] + h//2 + window_expand))
        
        # Find best matching position
        evaluator = MatchingEvaluator(self.Current_Image, current_kernel)
        best_score = -1.0
        best_pos = prev_center
        
        for x in range(x1_sw, x2_sw, step_size):
            for y in range(y1_sw, y2_sw, step_size):
                score = evaluator.evaluate_matching_score((x, y), (w, h))
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)
        
        # Check reliability and create new kernel
        if best_score < 0.7:  # ncc_threshold
            # Keep previous position if matching is unreliable
            best_pos = (int(prev_center[0] - w//2), int(prev_center[1] - h//2))
        
        # Extract new kernel
        x, y = best_pos
        new_kernel_content = self.Current_Image.image_content_GRB[y:y+h, x:x+w]
        new_pos_and_scale = (x, y, 1)
        new_kernel = CurrentKernel(new_kernel_content, new_pos_and_scale)
        
        # Template update if score is high enough
        if best_score >= 0.8:  # ncc_threshold + 0.1
            alpha = 0.2
            old_gray = current_kernel._to_gray().astype(np.float32)
            new_gray = new_kernel._to_gray().astype(np.float32)
            updated_gray = cv2.addWeighted(old_gray, 1-alpha, new_gray, alpha, 0).astype(np.uint8)
            updated_color = cv2.addWeighted(current_kernel.image_content_GRB.astype(np.float32), 1-alpha,
                                          new_kernel.image_content_GRB.astype(np.float32), alpha, 0).astype(np.uint8)
            new_kernel.image_content_GRB = updated_color
        
        return new_kernel, best_score
    
    def _weighted_velocity_predict(self):
        kernel_center_positions = [k.get_center_pos() for k in self.Kernel_Buffer.buffer[-self.length_for_prediction:]]
        kernel_center_positions_difference = []
        
        # Calculate velocity differences between consecutive frames
        for i in range(1, len(kernel_center_positions)):
            prev_pos = kernel_center_positions[i-1]
            curr_pos = kernel_center_positions[i]
            diff = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            kernel_center_positions_difference.append(diff)
        
        if len(kernel_center_positions_difference) == 0:
            return (0.0, 0.0)
        
        # Calculate weighted average for x and y velocities
        weight = 0.8  # Weight factor for exponential weighting
        x_velocities = [diff[0] for diff in kernel_center_positions_difference]
        y_velocities = [diff[1] for diff in kernel_center_positions_difference]
        
        weighted_vx = self._weighted_avg(x_velocities, weight)
        weighted_vy = self._weighted_avg(y_velocities, weight)
        
        return (weighted_vx, weighted_vy)
        
    def _weighted_avg(self, data, weight):
        if len(data) == 0:
            return 0.0
        
        if len(data) == 1:
            return data[0]

        weighted_sum = 0.0
        weight_sum = 0.0
        
        # More recent data gets higher weight
        for i, d in enumerate(data):
            w = weight ** (len(data) - 1 - i)
            weighted_sum += d * w
            weight_sum += w

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0


class KernelBuffer:
    def __init__(self):
        self.buffer = []
        
    def add_kernel(self, Current_Kernel):
        self.buffer.append(Current_Kernel)

def initialize_tracker(image_folder, init_frame_idx, init_bbox):
    img_paths = sorted(glob(os.path.join(image_folder, "cars_*.jpg")))
    img_paths = [p for p in img_paths if int(os.path.basename(p).split('_')[1].split('.')[0]) >= init_frame_idx]
    
    ## Create the first CurrentImage
    image_content_RGB_first = cv2.imread(img_paths[0])
    First_Img = CurrentImage(image_content_RGB_first)
    
    ## Create the first CurrentKernel
    x, y, w, h = init_bbox
    kernel_content_GRB_first = image_content_RGB_first[y:y+h, x:x+w]
    pos_and_scale_first = (x, y, 1) # No scale initially
    First_Kernel = CurrentKernel(kernel_content_GRB_first, pos_and_scale_first)
    
    ## Initialize KernelBuffer
    Kernel_Buffer = KernelBuffer()
    Kernel_Buffer.add_kernel(First_Kernel)

    ## Return the initialized entities
    return First_Kernel, Kernel_Buffer, img_paths

def visualization(Current_Image, Current_Kernel, score):
    frame = Current_Image.image_content_GRB.copy()
    x, y, scale = Current_Kernel.pos_and_scale
    w, h = Current_Kernel.get_size()
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow("Tracking", frame)
    key = cv2.waitKey(1)
    return key == ord('q')

if __name__ == "__main__":
    ## Define hyperparameters
    LENGTH_FOR_PREDICTION = 5
    
    ## Initialize the tracker for blue car sequence
    blue_init_frame = 1085
    blue_bbox = (677, 257, 163, 132)
    Current_Kernel, Kernel_Buffer, img_paths = initialize_tracker("./sequences/blue", blue_init_frame, blue_bbox)

    log = []
    
    # Start reading images and match searching
    for idx, current_img_path in enumerate(img_paths[1:], start=2):
        ## Get the current image
        Current_Image = CurrentImage(cv2.imread(current_img_path))
        
        ## Explore the next kernel
        Next_Kernel_Explorer = NextKernelExplore(Current_Image, Kernel_Buffer, LENGTH_FOR_PREDICTION)
        # Update the current kernel
        result = Next_Kernel_Explorer.explore_next_kernel()
        if result:
            Current_Kernel, best_score = result
            # Add the current kernel to buffer
            Kernel_Buffer.add_kernel(Current_Kernel)
            
            # Log results
            x, y, scale = Current_Kernel.pos_and_scale
            w, h = Current_Kernel.get_size()
            log.append({
                "frame_path": current_img_path,
                "bbox": (x, y, w, h),
                "score": best_score
            })
            
            ## Visualization
            if visualization(Current_Image, Current_Kernel, best_score):
                break
    
    cv2.destroyAllWindows()
    
    # Write log
    with open("blue_tracking_log.txt", 'w') as f:
        for x in log:
            f.write(f"{os.path.basename(x['frame_path'])}, {x['bbox'][0]}, {x['bbox'][1]}, {x['bbox'][2]}, {x['bbox'][3]}, {x['score']:.4f}\n")