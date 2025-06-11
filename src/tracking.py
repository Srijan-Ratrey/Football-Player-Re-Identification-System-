import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from pathlib import Path

def track_players(video_path, output_dir):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracking results
    tracking_results = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLOv8 tracking
        results = model.track(frame, persist=True, classes=[0])  # class 0 is person
        
        # Get tracking results
        frame_results = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                frame_results.append({
                    'id': int(track_id),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        tracking_results.append(frame_results)
        frame_count += 1
        
        # Save frame with tracking visualization
        if results[0].boxes is not None:
            annotated_frame = results[0].plot()
            cv2.imwrite(str(output_dir / f"frame_{frame_count:04d}.jpg"), annotated_frame)
    
    cap.release()
    
    # Save tracking results
    with open(output_dir / 'tracking_results.json', 'w') as f:
        json.dump(tracking_results, f)
    
    return tracking_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Track players in a video using YOLOv8')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('output_dir', help='Path to output directory')
    args = parser.parse_args()
    
    track_players(args.video_path, args.output_dir) 