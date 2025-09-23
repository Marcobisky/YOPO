import cv2
import numpy as np
import os
from glob import glob

class MatchingEvaluator:
    # Evaluate the matching score between current image and kernel
    def __init__(self, Current_Image, Current_Kernel):
        self.Current_Image = Current_Image
        self.Current_Kernel = Current_Kernel
        
    def 

class CurrentImage:
    # The current focused image
    def __init__(self, image_content_GRB):
        self.image_content_GRB = image_content_GRB

    def _to_gray(self):
        return cv2.cvtColor(self.image_content_GRB, cv2.COLOR_BGR2GRAY)
    
    def _get_size(self):


class CurrentKernel(CurrentImage):
    # The current kernel image used for pattern matching
    def __init__(self, kernel_content_GRB, pos_and_scale):
        super().__init__(kernel_content_GRB)
        self.pos_and_scale = pos_and_scale  # (x, y, scale)

class NextKernelExplore:
    # Explore the next kernel entity based on history
    def __init__(self, Kernel_Buffer, length_for_prediction=5):
        self.Kernel_Buffer = Kernel_Buffer
        self.length_for_prediction = length_for_prediction

    def explore_next_kernel(self):
        ...
        return Next_Kernel
        
    

class KernelBuffer:
    def __init__(self):
        self.buffer = []
        
    def add_kernel(self, Current_Kernel):
        self.buffer.append(Current_Kernel)

def initialize_tracker(image_folder, init_bbox):
    img_paths = sorted(glob(os.path.join(image_folder, "cars_*.jpg")))
    
    ## Create the first CurrentImage
    image_content_RGB_first = cv2.imread(img_paths[0])
    First_Img = CurrentImage(image_content_RGB_first)
    
    ## Create the first CurrentKernel
    kernel_content_GRB_first = image_content_RGB_first[init_bbox[1]:init_bbox[1]+init_bbox[3], init_bbox[0]:init_bbox[0]+init_bbox[2]]
    pos_and_scale_first = (init_bbox[0], init_bbox[1], 1) # No scale initially
    First_Kernel = CurrentKernel(kernel_content_GRB_first, pos_and_scale_first)
    
    ## Initialize KernelBuffer
    Kernel_Buffer = KernelBuffer()
    Kernel_Buffer.add_kernel(First_Kernel)

    ## Return the initialized entities
    return First_Kernel, Kernel_Buffer, img_paths


def visualization(...):
    ...


if __name__ == "__main__":
    ## Define hyperparameters
    LENGTH_FOR_PREDICTION = 5
    
    ## Initialize the tracker for blue car sequence
    Current_Kernel, Kernel_Buffer, img_paths = initialize_tracker("./sequences/blue", (677, 257, 163, 132))

    # Start reading images and match searching
    for current_img_path in img_paths[1:]:
        ## Get the current image
        Current_Image = CurrentImage(cv2.imread(current_img_path))
        
        ## Explore the next kernel
        Next_Kernel_Explorer = NextKernelExplore(Kernel_Buffer, LENGTH_FOR_PREDICTION)
        # Update the current kernel
        Current_Kernel = Next_Kernel_Explorer.explore_next_kernel()
        # Add the current kernel to buffer
        Kernel_Buffer.add_kernel(Current_Kernel)

        ## Visualization
        visualization(...)
        