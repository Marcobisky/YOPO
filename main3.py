import cv2
import numpy as np
import os
from glob import glob


def _extract_patch(Image, bbox):
    x, y, w, h = bbox
    Patch = CurrentKernel(Image.image_content_GRB[y:y+h, x:x+w], bbox)
    return Patch


def _ncc_score(Current_Image, Current_Kernel):
    ## Evaluate the matching score between current image and kernel
    # Extract image patch
    Patch_Image = _extract_patch(Current_Image, Current_Kernel.bbox)
    # Convert to gray
    patch_gray = Patch_Image.to_gray()
    kernel_gray = Current_Kernel.to_gray()
    # Use cv2.matchTemplate
    result = cv2.matchTemplate(patch_gray, kernel_gray, cv2.TM_CCOEFF_NORMED)
    # Return the maximum value and position
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc



def _compute_color_hist(Image, mask=None):
    # 使用 HSV 色彩直方图
    img = Image.image_content_GRB
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 只取 H 和 S 通道
    h_bins = 30
    s_bins = 32
    hist = cv2.calcHist([hsv], [0,1], mask, [h_bins, s_bins], [0,180, 0,256])
    cv2.normalize(hist, hist)
    return hist

def _is_color_similar(Image, Kernel_Image, Config):
    ## Evaluate the color similarity between two images
    # Extract image patch
    Patch_Image = _extract_patch(Image, Kernel_Image.bbox)
    hist1 = _compute_color_hist(Patch_Image)
    hist2 = _compute_color_hist(Kernel_Image)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return (similarity >= Config.color_threshold)


class CurrentImage:
    # The current focused image
    def __init__(self, image_content_GRB):
        self.image_content_GRB = image_content_GRB

    def to_gray(self):
        return cv2.cvtColor(self.image_content_GRB, cv2.COLOR_BGR2GRAY)
    
    def get_size(self):
        h, w = self.image_content_GRB.shape[:2]
        return (w, h)

class CurrentKernel(CurrentImage):
    # The current kernel image used for pattern matching
    def __init__(self, kernel_content_GRB, bbox, scale=1.0):
        super().__init__(kernel_content_GRB)
        self.bbox = bbox  # (x, y, width, height)
        self.scale = scale  # scale factor



class KernelBuffer:
    ## Store historical kernels, predict the most likely next kernel (cloud center) btw.
    def __init__(self, Config):
        self.buffer = []
        self.config = Config
        self.vx_avg = 0
        self.vy_avg = 0

    # def create_genesis_kernel(self, Genesis_Kernel):
    #     self.genesis_kernel = Genesis_Kernel
        
    def add_kernel(self, Current_Kernel):
        # First compute and update moving average velocity
        v_x_new = Current_Kernel.bbox[0] - self.buffer[-1].bbox[0] if len(self.buffer) > 0 else 0
        v_y_new = Current_Kernel.bbox[1] - self.buffer[-1].bbox[1] if len(self.buffer) > 0 else 0
        w = self.config.weight
        self.vx_avg = w * v_x_new + (1 - w) * self.vx_avg
        self.vy_avg = w * v_y_new + (1 - w) * self.vy_avg
        # Append the current kernel to the buffer
        self.buffer.append(Current_Kernel)
        
        
    
    def update_best_kernel(self, Best_Kernel):
        self.best_kernel = Best_Kernel
    
    def get_cloud(self):
        x_c, y_c = self._predict_cloud_center()
        # Define the searching space ("cloud") around the predicted kernel
        Cloud = []
        for x in range(int(x_c-self.config.pad_pixels), int(x_c+self.config.pad_pixels), int(self.config.step_size_pixels)):
            for y in range(int(y_c-self.config.pad_pixels), int(y_c+self.config.pad_pixels), int(self.config.step_size_pixels)):
                for sc in np.arange(self.best_kernel.scale-self.config.pad_scale, self.best_kernel.scale+self.config.pad_scale, self.config.step_size_scale):
                    # Compute stretched best kernel
                    stretched_kernel = cv2.resize(self.best_kernel.image_content_GRB, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                    w, h = stretched_kernel.shape[1], stretched_kernel.shape[0]
                    New_Kernel = CurrentKernel(stretched_kernel, (x, y, w, h), sc)
                    Cloud.append(New_Kernel)
        return Cloud
        
    def _predict_cloud_center(self):
        ## Predict the next most likely kernel ("cloud center") based on historical kernels
        x_c = self.buffer[-1].bbox[0] + self.vx_avg
        y_c = self.buffer[-1].bbox[1] + self.vy_avg
        return (x_c, y_c)

    # def _get_velocity(self):
    #     # Extract all x, y positions from historical kernels
    #     _x = [kernel.bbox[0] for kernel in self.buffer]
    #     _y = [kernel.bbox[1] for kernel in self.buffer]
    #     # Compute adjacent differences
    #     v_x = np.diff(_x)
    #     v_y = np.diff(_y)
    #     return v_x, v_y
    
    # def _moving_avg(self, data):
    #     # Compute the last length_for_prediction elements moving average
    #     denominator = sum([self.config.weight**i for i in range(self.config.length_for_prediction)])
    #     data_avg = 0
    #     for i in range(1, len(data)):
    #         if i < self.config.length_for_prediction:
    #             data_avg += self.config.weight**i * data[-i]
    #         else:
    #             break
    #     return data_avg / denominator

class Config:
    # Configuration for prediction
    def __init__(self, length_for_prediction, pad_pixels, step_size_pixels, pad_scale, step_size_scale, ncc_threshold, color_threshold, weight):
        self.length_for_prediction = length_for_prediction
        self.pad_pixels = pad_pixels
        self.step_size_pixels = step_size_pixels
        self.pad_scale = pad_scale
        self.step_size_scale = step_size_scale
        self.ncc_threshold = ncc_threshold
        self.color_threshold = color_threshold
        self.weight = weight # Moving average weight


def initialize_tracker(image_folder, init_bbox, Config):
    img_paths = sorted(glob(os.path.join(image_folder, "cars_*.jpg")))
    
    ## Create the first CurrentImage
    First_Img = CurrentImage(cv2.imread(img_paths[0]))
    
    ## Create the first CurrentKernel
    First_Kernel = _extract_patch(First_Img, init_bbox)
    
    ## Initialize KernelBuffer
    Kernel_Buffer = KernelBuffer(Config)
    # Kernel_Buffer.create_genesis_kernel(First_Kernel)
    Kernel_Buffer.add_kernel(First_Kernel)
    Kernel_Buffer.update_best_kernel(First_Kernel)

    ## Return the initialized entities
    return First_Kernel, Kernel_Buffer, img_paths


def update_next_kernel(Current_Image, Kernel_Buffer, Config):
    Best_Next_Kernel = Kernel_Buffer.best_kernel
    best_score = -1
    ## Get the next kernel based on traverse matching
    for Kernel_Test in Kernel_Buffer.get_cloud():        
        # Use color histogram similarity as a preliminary filter
        if not _is_color_similar(Current_Image, Kernel_Test, Config):
            continue
        
        # Check NCC Matching
        score, _ = _ncc_score(Current_Image, Kernel_Test)
        if score > best_score:
            best_score = score
            Best_Next_Kernel = _extract_patch(Current_Image, Kernel_Test.bbox)
        
    Kernel_Buffer.add_kernel(Best_Next_Kernel)
        
    # Whether to update the best kernel
    if best_score > Config.ncc_threshold:
        Kernel_Buffer.update_best_kernel(Best_Next_Kernel)
        print("Best kernel updated.")
        print(Best_Next_Kernel.bbox)
    
    return Best_Next_Kernel, best_score

def visualization(Current_Image, Best_Kernel, score):
    frame = Current_Image.image_content_GRB.copy()
    
    # 获取边框信息
    x, y, w, h = Best_Kernel.bbox
    
    # 在图像上画边框
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 添加得分文本
    cv2.putText(frame, f"{score:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 显示图像
    cv2.imshow("Tracking", frame)
    
    # 检查退出键
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        return True  # 返回True表示需要退出
    
    return False


if __name__ == "__main__":
    ## Define hyperparameters
    CONFIG = Config(length_for_prediction=8, 
                    pad_pixels=3, 
                    step_size_pixels=1, 
                    pad_scale=0.004, 
                    step_size_scale=0.002, 
                    ncc_threshold=0.95, 
                    color_threshold=0.8,
                    weight=0.2)
    
    # ## Initialize the tracker for blue car sequence
    # Current_Kernel, Kernel_Buffer, img_paths = initialize_tracker("./sequences/blue", (672, 255, 168, 132), CONFIG)

    # # Start reading images and match searching
    # for current_img_path in img_paths[1:]:
    #     ## Get the current image
    #     Current_Image = CurrentImage(cv2.imread(current_img_path))
        
    #     ## update the next best kernel
    #     Best_Next_Kernel, best_score = update_next_kernel(Current_Image, Kernel_Buffer, CONFIG)
        
    #     ## Print messages
    #     print(f"Processing {os.path.basename(current_img_path)}: Best Score = {best_score:.4f}")
        
    #     ## Visualization
    #     should_exit = visualization(Current_Image, Best_Next_Kernel, best_score)
    #     if should_exit:
    #         break
    
    # # 清理窗口
    # cv2.destroyAllWindows()
        
        
        
        
    ## Define hyperparameters
    CONFIG = Config(length_for_prediction=8, 
                    pad_pixels=3, 
                    step_size_pixels=1, 
                    pad_scale=0.004, 
                    step_size_scale=0.002, 
                    ncc_threshold=0.97, 
                    color_threshold=0.8,
                    weight=0.8) 
    ## Initialize the tracker for red car sequence
    Current_Kernel, Kernel_Buffer, img_paths = initialize_tracker("./sequences/red", (796, 266, 196, 151), CONFIG)
    
   # Start reading images and match searching
    for current_img_path in img_paths[1:]:
        ## Get the current image
        Current_Image = CurrentImage(cv2.imread(current_img_path))
        
        ## update the next best kernel
        Best_Next_Kernel, best_score = update_next_kernel(Current_Image, Kernel_Buffer, CONFIG)
        
        ## Print messages
        print(f"Processing {os.path.basename(current_img_path)}: Best Score = {best_score:.4f}")
        
        ## Visualization
        should_exit = visualization(Current_Image, Best_Next_Kernel, best_score)
        if should_exit:
            break
    
    # 清理窗口
    cv2.destroyAllWindows() 