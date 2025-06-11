import os
import cv2
import numpy as np
from pathlib import Path
from tracking import track_players
from visualization import visualize_tracking

def process_video(video_path, output_dir):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track players
    print("Tracking players...")
    tracking_results = track_players(video_path, output_dir)
    
    # Visualize tracking
    print("Visualizing tracking...")
    visualize_tracking(video_path, tracking_results, output_dir)
    
    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process a football video for player tracking')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--output_dir', default='results', help='Path to output directory')
    args = parser.parse_args()
    
    process_video(args.video_path, args.output_dir) 