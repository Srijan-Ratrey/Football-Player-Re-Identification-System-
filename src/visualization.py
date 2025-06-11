import cv2
import numpy as np
from pathlib import Path

def visualize_tracking(video_path, tracking_results, output_dir):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = Path(output_dir) / 'tracking_visualization.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw tracking results
        if frame_count < len(tracking_results):
            for track in tracking_results[frame_count]:
                x1, y1, x2, y2 = track['bbox']
                track_id = track['id']
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw track ID
                cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return output_path 