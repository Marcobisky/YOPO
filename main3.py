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
        self.best_score = -1

    def create_genesis_kernel(self, Genesis_Kernel):
        self.genesis_kernel = Genesis_Kernel
        
    def add_kernel(self, Current_Kernel):
        self.buffer.append(Current_Kernel)
    
    def update_best_kernel(self, Best_Kernel):
        self.best_kernel = Best_Kernel
    
    def get_cloud(self):
        x_c, y_c = self._predict_cloud_center()
        # Define the searching space ("cloud") around the predicted kernel
        Cloud = []
        ...
        for delta_x in range(..., ..., self.config.step_size):
            for delta_y in range(..., ..., self.config.step_size):
                for sc in range(..., ..., self.config.step_size_scale):
                    # Compute stretched best kernel
                    Stretched_Kernel = cv2.resize(...)
                    # Compute bbox
                    x = x_c + delta_x
                    y = y_c + delta_y
                    w, h = Stretched_Kernel.get_size()
                    New_Kernel = CurrentKernel(Stretched_Kernel, (x, y, w, h), sc)
                    Cloud.append(New_Kernel)
        return Cloud
        
    def _predict_cloud_center(self):
        # Predict the next most likely kernel ("cloud center") based on historical kernels
        ...
        return (x_c, y_c)

class Config:
    # Configuration for prediction
    def __init__(self, length_for_prediction, pad_pixels, step_size_pixels, pad_scale, step_size_scale, ncc_threshold, color_threshold, alpha):
        self.length_for_prediction = length_for_prediction
        self.pad_pixels = pad_pixels
        self.step_size_pixels = step_size_pixels
        self.pad_scale = pad_scale
        self.step_size_scale = step_size_scale
        self.ncc_threshold = ncc_threshold
        self.color_threshold = color_threshold
        self.alpha = alpha


def initialize_tracker(image_folder, init_bbox, Config):
    img_paths = sorted(glob(os.path.join(image_folder, "cars_*.jpg")))
    
    ## Create the first CurrentImage
    First_Img = CurrentImage(cv2.imread(img_paths[0]))
    
    ## Create the first CurrentKernel
    First_Kernel = _extract_patch(First_Img, init_bbox)
    
    ## Initialize KernelBuffer
    Kernel_Buffer = KernelBuffer(Config)
    Kernel_Buffer.create_genesis_kernel(First_Kernel)
    Kernel_Buffer.update_best_kernel(First_Kernel)

    ## Return the initialized entities
    return First_Kernel, Kernel_Buffer, img_paths


def get_next_kernel(Current_Image, Kernel_Buffer, Config):
    ## Get the next kernel based on traverse matching
    for Kernel_Test in Kernel_Buffer.get_cloud():        
        # Use color histogram similarity as a preliminary filter
        if not _is_color_similar(Current_Image, Kernel_Test, Config):
            continue
        
        # Check NCC Matching
        score, _ = _ncc_score(Current_Image, Kernel_Test)
        if score > Kernel_Buffer.best_score:
            Kernel_Buffer.best_score = score
            ...
        
        return Best_Next_Kernel

def visualization(...):
    ...


if __name__ == "__main__":
    ## Define hyperparameters
    CONFIG = Config(5, 50, 5, 1.2, 0.05)
    
    ## Initialize the tracker for blue car sequence
    Current_Kernel, Kernel_Buffer, img_paths = initialize_tracker("./sequences/blue", (677, 257, 163, 132), CONFIG)

    # Start reading images and match searching
    for current_img_path in img_paths[1:]:
        ## Get the current image
        Current_Image = CurrentImage(cv2.imread(current_img_path))
        
        ## Explore and update the next best kernel
        Current_Kernel = get_next_kernel(Current_Image, Current_Kernel, CONFIG)
        # Add the current kernel to buffer
        Kernel_Buffer.add_kernel(Current_Kernel)

        ## Visualization
        visualization(...)
        