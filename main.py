import cv2
import numpy as np
import os
from glob import glob
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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
    # Use HSV histogram
    img = Image.image_content_GRB
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Only take H and S channels
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
                for sc in np.arange(1-self.config.pad_scale, 1+self.config.pad_scale, self.config.step_size_scale):
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

class TrackingLogger:
    # Logger for tracking results and parameters
    def __init__(self, config, sequence_type):
        self.config = config
        self.sequence_type = sequence_type  # "blue" or "red"
        self.data = {
            "config": self._config_to_dict(),
            "sequence_type": sequence_type,
            "frames": []
        }
        self.log_filename = self._generate_log_filename()
        
    def _config_to_dict(self):
        # Convert config object to dictionary for JSON serialization
        return {
            "length_for_prediction": self.config.length_for_prediction,
            "pad_pixels": self.config.pad_pixels,
            "step_size_pixels": self.config.step_size_pixels,
            "pad_scale": self.config.pad_scale,
            "step_size_scale": self.config.step_size_scale,
            "ncc_threshold": self.config.ncc_threshold,
            "color_threshold": self.config.color_threshold,
            "weight": self.config.weight
        }
    
    def _generate_log_filename(self):
        # Generate log filename with current date and sequence type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"logs/{timestamp}_{self.sequence_type}.json"
    
    def log_frame(self, frame_name, best_score, best_kernel_bbox):
        # Log data for a single frame
        x, y, w, h = best_kernel_bbox
        area = w * h
        
        frame_data = {
            "frame_name": frame_name,
            "best_score": best_score,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": w,
            "bbox_height": h,
            "bbox_area": area
        }
        self.data["frames"].append(frame_data)
    
    def save_log(self):
        # Save all logged data to JSON file
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.log_filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Log saved to {self.log_filename}")


class SearchSpaceVisualizer:
    # Visualizer for 2D search space at best kernel scale
    def __init__(self, config, title="Search Space Visualization"):
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Create a better colormap for scores in 0.7-1.0 range
        colors = ['#000080', '#0000FF', '#4169E1', '#00BFFF', '#00FFFF', 
                  '#00FF00', '#FFFF00', '#FFA500', '#FF4500', '#FF0000']
        self.colormap = LinearSegmentedColormap.from_list('high_contrast', colors)
        
        self.initialized = False
        
        # Fixed grid parameters
        self.pad_pixels = config.pad_pixels
        self.step_size = config.step_size_pixels
        
        # Create fixed coordinate arrays
        self.fixed_x_range = np.arange(-self.pad_pixels, self.pad_pixels + self.step_size, self.step_size)
        self.fixed_y_range = np.arange(-self.pad_pixels, self.pad_pixels + self.step_size, self.step_size)
        
        # Fixed extent (never changes)
        self.fixed_extent = (-self.pad_pixels, self.pad_pixels, self.pad_pixels, -self.pad_pixels)
        
    def update_search_space(self, current_image, kernel_buffer, best_kernel, best_score, frame_name=None):
        # Get predicted center (for actual search positions)
        if len(kernel_buffer.buffer) > 0:
            x_c, y_c = kernel_buffer._predict_cloud_center()
        else:
            x_c, y_c = best_kernel.bbox[0], best_kernel.bbox[1]
        
        # Initialize fixed score grid
        score_grid = np.zeros((len(self.fixed_y_range), len(self.fixed_x_range)))
        
        # Evaluate scores at each position in the FIXED 2D search space
        for i, y_offset in enumerate(self.fixed_y_range):
            for j, x_offset in enumerate(self.fixed_x_range):
                # Calculate actual position
                actual_x = int(x_c + x_offset)
                actual_y = int(y_c + y_offset)
                
                # Create kernel at current position with best scale
                stretched_kernel = cv2.resize(best_kernel.image_content_GRB, None, 
                                            fx=best_kernel.scale, fy=best_kernel.scale, 
                                            interpolation=cv2.INTER_CUBIC)
                w, h = stretched_kernel.shape[1], stretched_kernel.shape[0]
                
                # Check bounds
                if (actual_x >= 0 and actual_y >= 0 and 
                    actual_x + w < current_image.image_content_GRB.shape[1] and 
                    actual_y + h < current_image.image_content_GRB.shape[0]):
                    
                    test_kernel = CurrentKernel(stretched_kernel, (actual_x, actual_y, w, h), best_kernel.scale)
                    
                    # Compute NCC score
                    ncc_score, _ = _ncc_score(current_image, test_kernel)
                    
                    # Compute color similarity
                    color_similar = _is_color_similar(current_image, test_kernel, self.config)
                    
                    # Combined score
                    if color_similar:
                        score_grid[i, j] = ncc_score
                    else:
                        score_grid[i, j] = ncc_score * 0.5  # Penalize color mismatch
                else:
                    score_grid[i, j] = 0  # Out of bounds
        
        # Update visualization
        self.ax.clear()
        
        # Display the score heatmap with FIXED extent and enhanced color range for 0.7-1.0
        im = self.ax.imshow(score_grid, cmap=self.colormap, aspect='equal', 
                           extent=self.fixed_extent, interpolation='bilinear',
                           vmin=0.7, vmax=1.0, alpha=0.9)  # Focus color range on high scores
        
        # Mark the best position in FIXED coordinate system
        # Find the best score position in the grid to mark correctly
        max_score_idx = np.unravel_index(np.argmax(score_grid), score_grid.shape)
        best_grid_y, best_grid_x = max_score_idx
        
        # Convert grid indices to offset coordinates
        if best_grid_y < len(self.fixed_y_range) and best_grid_x < len(self.fixed_x_range):
            best_offset_x = self.fixed_x_range[best_grid_x]
            best_offset_y = self.fixed_y_range[best_grid_y]
            
            # Use cross marker instead of star
            self.ax.plot(best_offset_x, best_offset_y, 'white', marker='+', markersize=25, 
                        markeredgecolor='black', markeredgewidth=3, 
                        label=f'Best Match (Score: {best_score:.3f})', zorder=10)
        
        # Set FIXED axis limits
        self.ax.set_xlim(-self.pad_pixels, self.pad_pixels)
        self.ax.set_ylim(-self.pad_pixels, self.pad_pixels)
        
        # Set labels and title
        self.ax.set_xlabel('X Offset (pixels)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Offset (pixels)', fontsize=12, fontweight='bold')
        self.ax.set_title(f'2D Search Space at Scale {best_kernel.scale:.3f}', 
                         fontsize=12, fontweight='bold')
        
        # Add frame information as text annotation on the plot
        if frame_name:
            # Extract frame number from filename like 'cars_1085.jpg'
            try:
                frame_number = frame_name.split('_')[1].split('.')[0]
                self.ax.text(0.02, 0.98, f'Frame: {frame_number}', 
                            transform=self.ax.transAxes, fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                            verticalalignment='top', zorder=15)
            except (IndexError, ValueError):
                # If frame name parsing fails, just show the full name
                self.ax.text(0.02, 0.98, f'Frame: {frame_name}', 
                            transform=self.ax.transAxes, fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                            verticalalignment='top', zorder=15)
        
        # Add colorbar if not initialized
        if not self.initialized:
            cbar = plt.colorbar(im, ax=self.ax, shrink=0.8, pad=0.02)
            cbar.set_label('Matching Score', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            self.initialized = True
        
        # Add legend with improved styling
        legend = self.ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                               framealpha=0.9, edgecolor='black')
        legend.get_frame().set_facecolor('white')
        
        # Add grid with improved styling
        self.ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        
        # Set background color
        self.ax.set_facecolor('#f8f9fa')
        
        # Update display
        plt.pause(0.001)
        plt.draw()


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
            Best_Next_Kernel.scale = Kernel_Test.scale  # Preserve scale information
        
    Kernel_Buffer.add_kernel(Best_Next_Kernel)
        
    # Whether to update the best kernel
    if best_score > Config.ncc_threshold:
        Kernel_Buffer.update_best_kernel(Best_Next_Kernel)
        print("Best kernel updated.")
        print(f"Best kernel bbox: {Best_Next_Kernel.bbox}, scale: {Best_Next_Kernel.scale:.4f}")
    
    return Best_Next_Kernel, best_score







def visualization(Current_Image, Best_Kernel, score, trajectory_points=None):
    frame = Current_Image.image_content_GRB.copy()
    x, y, w, h = Best_Kernel.bbox
    
    # Draw rectangle on the image
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add score text
    cv2.putText(frame, f"{score:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw trajectory if points are provided
    if trajectory_points and len(trajectory_points) > 1:
        # Convert trajectory points to numpy array for drawing
        points = np.array(trajectory_points, dtype=np.int32)
        
        # Draw trajectory line
        for i in range(1, len(points)):
            cv2.line(frame, tuple(points[i-1]), tuple(points[i]), (255, 0, 0), 2)
        
        # Draw trajectory points
        for point in points:
            cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
        
        # Highlight current position
        current_center = (x + w//2, y + h//2)
        cv2.circle(frame, current_center, 5, (255, 255, 0), -1)
    
    # Show now
    cv2.imshow("Tracking", frame)
    
    # Check for 'q' key to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        return True  # True for exit
    
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
    
    ## Initialize the tracker for blue car sequence
    Current_Kernel, Kernel_Buffer, img_paths = initialize_tracker("./sequences/blue", (672, 255, 168, 132), CONFIG)

    # Initialize logger for blue sequence
    logger = TrackingLogger(CONFIG, "blue")
    
    # Initialize search space visualizer
    visualizer = SearchSpaceVisualizer(CONFIG, "Blue Car Tracking - Search Space")
    
    # Initialize trajectory tracking
    trajectory_points = []
    # Add initial point from first kernel
    x0, y0, w0, h0 = Current_Kernel.bbox
    trajectory_points.append((x0 + w0//2, y0 + h0//2))

    # Start reading images and match searching
    for current_img_path in img_paths[1:]:
        ## Get the current image
        Current_Image = CurrentImage(cv2.imread(current_img_path))
        
        ## update the next best kernel
        Best_Next_Kernel, best_score = update_next_kernel(Current_Image, Kernel_Buffer, CONFIG)
        
        ## Update search space visualization with frame information
        visualizer.update_search_space(Current_Image, Kernel_Buffer, Best_Next_Kernel, best_score, 
                                     os.path.basename(current_img_path))
        
        ## Update trajectory with bounding box center
        x, y, w, h = Best_Next_Kernel.bbox
        center_x, center_y = x + w//2, y + h//2
        trajectory_points.append((center_x, center_y))
        
        ## Log frame data
        logger.log_frame(os.path.basename(current_img_path), best_score, Best_Next_Kernel.bbox)
        
        ## Print messages
        print(f"Processing {os.path.basename(current_img_path)}: Best Score = {best_score:.4f}")
        
        ## Visualization with trajectory (OpenCV window)
        should_exit = visualization(Current_Image, Best_Next_Kernel, best_score, trajectory_points)
        if should_exit:
            break
    
    # Save log before exiting
    logger.save_log()
    
    # Clear windows
    cv2.destroyAllWindows()
    plt.close('all')
        
        
        
        
    # ## Define hyperparameters
    # CONFIG = Config(length_for_prediction=8, 
    #                 pad_pixels=3, 
    #                 step_size_pixels=1, 
    #                 pad_scale=0.004, 
    #                 step_size_scale=0.002, 
    #                 ncc_threshold=0.97, 
    #                 color_threshold=0.8,
    #                 weight=0.8) 
    # ## Initialize the tracker for red car sequence
    # Current_Kernel, Kernel_Buffer, img_paths = initialize_tracker("./sequences/red", (796, 266, 196, 151), CONFIG)
    
    # # Initialize logger for red sequence
    # logger = TrackingLogger(CONFIG, "red")
    
    # # Initialize search space visualizer
    # visualizer = SearchSpaceVisualizer(CONFIG, "Red Car Tracking - Search Space")
    
    # # Initialize trajectory tracking
    # trajectory_points = []
    # # Add initial point from first kernel
    # x0, y0, w0, h0 = Current_Kernel.bbox
    # trajectory_points.append((x0 + w0//2, y0 + h0//2))
    
    # # Start reading images and match searching
    # for current_img_path in img_paths[1:]:
    #     ## Get the current image
    #     Current_Image = CurrentImage(cv2.imread(current_img_path))
        
    #     ## update the next best kernel
    #     Best_Next_Kernel, best_score = update_next_kernel(Current_Image, Kernel_Buffer, CONFIG)
        
    #     ## Update search space visualization with frame information
    #     visualizer.update_search_space(Current_Image, Kernel_Buffer, Best_Next_Kernel, best_score, 
    #                                  os.path.basename(current_img_path))
        
    #     ## Update trajectory with bounding box center
    #     x, y, w, h = Best_Next_Kernel.bbox
    #     center_x, center_y = x + w//2, y + h//2
    #     trajectory_points.append((center_x, center_y))
        
    #     ## Log frame data
    #     logger.log_frame(os.path.basename(current_img_path), best_score, Best_Next_Kernel.bbox)
        
    #     ## Print messages
    #     print(f"Processing {os.path.basename(current_img_path)}: Best Score = {best_score:.4f}")
        
    #     ## Visualization with trajectory (OpenCV window)
    #     should_exit = visualization(Current_Image, Best_Next_Kernel, best_score, trajectory_points)
    #     if should_exit:
    #         break
    
    # # Save log before exiting
    # logger.save_log()
    
    # # Clear windows
    # cv2.destroyAllWindows()
    # plt.close('all')
        
        
    