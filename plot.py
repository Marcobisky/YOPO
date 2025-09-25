#!/usr/bin/env python3
"""
Plot tracking results from log files.
This script reads JSON log files from the logs folder and creates visualizations
for tracking performance metrics across frames.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def load_log_data(log_folder="logs"):
    """
    Load all log files from the specified folder.
    
    Args:
        log_folder (str): Path to the folder containing log files
        
    Returns:
        dict: Dictionary containing loaded data organized by sequence type
    """
    log_files = glob(os.path.join(log_folder, "*.json"))
    data = {}
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
            
        # Extract sequence type and timestamp from filename
        filename = os.path.basename(log_file)
        timestamp = filename.split('_')[0] + '_' + filename.split('_')[1]
        sequence_type = log_data['sequence_type']
        
        # Store data with a unique identifier
        key = f"{timestamp}_{sequence_type}"
        data[key] = log_data
        
    return data


def extract_frame_numbers(frame_names):
    """
    Extract frame numbers from frame names (e.g., 'cars_1085.jpg' -> 1085).
    
    Args:
        frame_names (list): List of frame names
        
    Returns:
        list: List of frame numbers
    """
    frame_numbers = []
    for name in frame_names:
        # Extract number from filename like 'cars_1085.jpg'
        try:
            number = int(name.split('_')[1].split('.')[0])
            frame_numbers.append(number)
        except (IndexError, ValueError):
            # If extraction fails, use sequential numbering
            frame_numbers.append(len(frame_numbers) + 1)
    
    return frame_numbers


def filter_unique_configs(data):
    """
    Filter out duplicate configurations, keeping only the most recent one for each unique config.
    
    Args:
        data (dict): Dictionary containing loaded log data
        
    Returns:
        dict: Filtered data with unique configurations
    """
    config_groups = {}
    
    # Group by configuration
    for key, log_data in data.items():
        config = log_data['config']
        sequence_type = log_data['sequence_type']
        
        # Create a config signature
        config_signature = (
            config['length_for_prediction'],
            config['pad_pixels'],
            config['step_size_pixels'],
            config['pad_scale'],
            config['step_size_scale'],
            config['ncc_threshold'],
            config['color_threshold'],
            config['weight'],
            sequence_type
        )
        
        if config_signature not in config_groups:
            config_groups[config_signature] = []
        
        # Extract timestamp for sorting
        timestamp = key.split('_')[0] + '_' + key.split('_')[1]
        config_groups[config_signature].append((timestamp, key, log_data))
    
    # Keep only the most recent log for each config
    filtered_data = {}
    for config_signature, logs in config_groups.items():
        # Sort by timestamp (most recent first)
        logs.sort(key=lambda x: x[0], reverse=True)
        # Keep the most recent one
        _, key, log_data = logs[0]
        filtered_data[key] = log_data
    
    return filtered_data


def get_color_by_date(timestamp, all_timestamps, base_colors):
    """
    Generate color based on date (more recent = darker).
    
    Args:
        timestamp (str): Current timestamp
        all_timestamps (list): All timestamps sorted
        base_colors (list): Base color palette
        
    Returns:
        tuple: RGB color tuple
    """
    # Get position in sorted list (0 = oldest, len-1 = newest)
    position = all_timestamps.index(timestamp)
    total_count = len(all_timestamps)
    
    # Calculate intensity (0.3 to 1.0, where 1.0 is darkest/most recent)
    intensity = 0.3 + 0.7 * (position / max(1, total_count - 1))
    
    # Get base color
    color_idx = position % len(base_colors)
    base_color = base_colors[color_idx]
    
    # Convert color name to RGB and apply intensity
    color_map = {
        'blue': (0, 0, 1),
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'orange': (1, 0.5, 0),
        'purple': (0.5, 0, 0.5),
        'brown': (0.6, 0.3, 0)
    }
    
    if base_color in color_map:
        r, g, b = color_map[base_color]
        return (r * intensity, g * intensity, b * intensity)
    else:
        return (intensity, intensity, intensity)


def plot_tracking_results(data):
    """
    Create plots for tracking results.
    
    Args:
        data (dict): Dictionary containing loaded log data
    """
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        # Fallback to available style if seaborn is not available
        plt.style.use('default')
    
    # Filter unique configurations
    filtered_data = filter_unique_configs(data)
    
    # Create two separate figures
    fig1 = plt.figure(figsize=(14, 8))
    ax1 = fig1.add_subplot(111)
    fig1.suptitle('Best Matching Score vs Frame', fontsize=16, fontweight='bold')
    
    fig2 = plt.figure(figsize=(14, 8))
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('Bounding Box Area vs Frame', fontsize=16, fontweight='bold')
    
    base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Sort data by timestamp for color assignment
    sorted_items = sorted(filtered_data.items(), key=lambda x: x[0].split('_')[0] + '_' + x[0].split('_')[1])
    all_timestamps = [key.split('_')[0] + '_' + key.split('_')[1] for key, _ in sorted_items]
    
    # Prepare data for plotting
    for key, log_data in sorted_items:
        frames = log_data['frames']
        sequence_type = log_data['sequence_type']
        config = log_data['config']
        
        # Extract frame numbers and metrics
        frame_names = [frame['frame_name'] for frame in frames]
        frame_numbers = extract_frame_numbers(frame_names)
        best_scores = [frame['best_score'] for frame in frames]
        bbox_areas = [frame['bbox_area'] for frame in frames]
        
        # Get timestamp and color
        timestamp = key.split('_')[0] + '_' + key.split('_')[1]
        color = get_color_by_date(timestamp, all_timestamps, base_colors)
        
        # Create label with ncc_threshold
        label = f"{sequence_type.capitalize()} - {timestamp} (NCC={config['ncc_threshold']:.2f})"
        
        # Plot Best Score vs Frame
        ax1.plot(frame_numbers, best_scores, 
                color=color, marker='o', markersize=4, 
                linewidth=2, label=label, alpha=0.8)
        
        # Plot Bbox Area vs Frame
        ax2.plot(frame_numbers, bbox_areas, 
                color=color, marker='s', markersize=4, 
                linewidth=2, label=label, alpha=0.8)
    
    # Configure Best Score plot
    ax1.set_title('Best Matching Score vs Frame (with NCC Threshold)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Best Score', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    # Configure Bbox Area plot
    ax2.set_title('Bounding Box Area vs Frame', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('Bounding Box Area (pixels²)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    # Adjust layout for both figures
    fig1.tight_layout()
    fig2.tight_layout()
    
    # Save both plots
    fig1.savefig('best_score_analysis.png', dpi=300, bbox_inches='tight')
    fig2.savefig('bbox_area_analysis.png', dpi=300, bbox_inches='tight')
    
    print(f"Plots saved as 'best_score_analysis.png' and 'bbox_area_analysis.png'")
    print(f"Plotted {len(filtered_data)} unique configurations (duplicates filtered out)")
    
    # Show both plots in separate windows
    plt.show()


def print_summary_statistics(data):
    """
    Print summary statistics for all logged runs.
    
    Args:
        data (dict): Dictionary containing loaded log data
    """
    print("\n" + "="*60)
    print("TRACKING PERFORMANCE SUMMARY")
    print("="*60)
    
    for key, log_data in data.items():
        frames = log_data['frames']
        sequence_type = log_data['sequence_type']
        config = log_data['config']
        
        # Calculate statistics
        best_scores = [frame['best_score'] for frame in frames]
        bbox_areas = [frame['bbox_area'] for frame in frames]
        
        avg_score = np.mean(best_scores)
        min_score = np.min(best_scores)
        max_score = np.max(best_scores)
        std_score = np.std(best_scores)
        
        avg_area = np.mean(bbox_areas)
        std_area = np.std(bbox_areas)
        
        print(f"\nRun: {key}")
        print(f"Sequence Type: {sequence_type.capitalize()}")
        print(f"Number of Frames: {len(frames)}")
        print(f"Config - NCC Threshold: {config['ncc_threshold']}")
        print(f"Config - Color Threshold: {config['color_threshold']}")
        print(f"Config - Weight: {config['weight']}")
        print(f"Score Statistics:")
        print(f"  Average: {avg_score:.4f}")
        print(f"  Min: {min_score:.4f}")
        print(f"  Max: {max_score:.4f}")
        print(f"  Std Dev: {std_score:.4f}")
        print(f"Area Statistics:")
        print(f"  Average: {avg_area:.2f} pixels²")
        print(f"  Std Dev: {std_area:.2f} pixels²")
        print("-" * 40)


def main():
    """
    Main function to load data and create visualizations.
    """
    # Load log data
    data = load_log_data()
    
    if not data:
        print("No log files found in the 'logs' folder.")
        print("Please run main.py first to generate tracking logs.")
        return
    
    print(f"Found {len(data)} log file(s)")
    
    # Print summary statistics
    print_summary_statistics(data)
    
    # Create plots
    plot_tracking_results(data)


if __name__ == "__main__":
    main()