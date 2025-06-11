import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Load video file and return VideoCapture object.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        cv2.VideoCapture: Video capture object
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    return cap

def get_video_properties(cap: cv2.VideoCapture) -> Dict:
    """
    Get video properties.
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        
    Returns:
        Dict: Video properties including fps, frame count, width, height
    """
    properties = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0
    }
    
    properties['duration'] = properties['frame_count'] / properties['fps']
    return properties

def create_video_writer(output_path: str, fps: float, width: int, height: int, 
                       codec: str = 'mp4v') -> cv2.VideoWriter:
    """
    Create video writer for output.
    
    Args:
        output_path (str): Output video path
        fps (float): Frames per second
        width (int): Frame width
        height (int): Frame height
        codec (str): Video codec
        
    Returns:
        cv2.VideoWriter: Video writer object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise ValueError(f"Could not create video writer: {output_path}")
    
    return writer

def draw_tracks(frame: np.ndarray, tracks: List[Dict], 
               colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Draw tracking results on frame.
    
    Args:
        frame (np.ndarray): Input frame
        tracks (List[Dict]): List of track information
        colors (Optional[List[Tuple]]): Colors for different tracks
        
    Returns:
        np.ndarray: Frame with drawn tracks
    """
    if colors is None:
        # Generate random colors for tracks
        np.random.seed(42)  # For consistent colors
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), 
                  np.random.randint(0, 255)) for _ in range(100)]
    
    output_frame = frame.copy()
    
    for track in tracks:
        track_id = track['track_id']
        bbox = track['bbox']
        confidence = track.get('confidence', 0.0)
        
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[track_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID and confidence
        label = f"ID: {track_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text
        cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(output_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return output_frame

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_tracking_results(tracks_data: List[List[Dict]], output_path: str):
    """
    Save tracking results to JSON file.
    
    Args:
        tracks_data (List[List[Dict]]): Tracking results
        output_path (str): Output file path
    """
    # Convert numpy types to Python native types recursively
    serializable_data = convert_numpy(tracks_data)
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Tracking results saved to: {output_path}")

def load_tracking_results(input_path: str) -> List[List[Dict]]:
    """
    Load tracking results from JSON file.
    
    Args:
        input_path (str): Input JSON file path
        
    Returns:
        List[List[Dict]]: Loaded tracking results
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert feature lists back to numpy arrays
    for frame_tracks in data:
        for track in frame_tracks:
            if 'feature' in track:
                track['feature'] = np.array(track['feature'])
    
    return data

def visualize_track_statistics(tracks_data: List[List[Dict]], output_dir: str):
    """
    Create visualizations of tracking statistics.
    
    Args:
        tracks_data (List[List[Dict]]): Tracking results for each frame
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect statistics
    track_lengths = {}
    tracks_per_frame = []
    unique_track_ids = set()
    
    for frame_idx, frame_tracks in enumerate(tracks_data):
        tracks_per_frame.append(len(frame_tracks))
        
        for track in frame_tracks:
            track_id = track['track_id']
            unique_track_ids.add(track_id)
            
            if track_id not in track_lengths:
                track_lengths[track_id] = 0
            track_lengths[track_id] += 1
    
    # Plot 1: Tracks per frame
    plt.figure(figsize=(12, 6))
    plt.plot(tracks_per_frame)
    plt.title('Number of Tracked Players per Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Tracks')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'tracks_per_frame.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Track length distribution
    plt.figure(figsize=(10, 6))
    lengths = list(track_lengths.values())
    plt.hist(lengths, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Track Lengths')
    plt.xlabel('Track Length (frames)')
    plt.ylabel('Number of Tracks')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'track_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Track timeline
    plt.figure(figsize=(15, 8))
    track_ids = sorted(unique_track_ids)
    
    for i, track_id in enumerate(track_ids):
        frames_with_track = []
        for frame_idx, frame_tracks in enumerate(tracks_data):
            if any(track['track_id'] == track_id for track in frame_tracks):
                frames_with_track.append(frame_idx)
        
        if frames_with_track:
            plt.scatter(frames_with_track, [i] * len(frames_with_track), 
                       alpha=0.6, s=10)
    
    plt.title('Track Timeline')
    plt.xlabel('Frame Number')
    plt.ylabel('Track ID')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'track_timeline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    stats = {
        'total_frames': len(tracks_data),
        'unique_tracks': len(unique_track_ids),
        'avg_tracks_per_frame': np.mean(tracks_per_frame),
        'max_tracks_per_frame': max(tracks_per_frame),
        'avg_track_length': np.mean(lengths),
        'max_track_length': max(lengths) if lengths else 0,
        'min_track_length': min(lengths) if lengths else 0
    }
    
    with open(os.path.join(output_dir, 'tracking_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Tracking statistics saved to: {output_dir}")

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame (np.ndarray): Input frame
        target_size (Tuple[int, int]): Target size (width, height)
        
    Returns:
        np.ndarray: Resized frame
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create padded frame if needed
    if new_w != target_w or new_h != target_h:
        # Create black background
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized frame
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return padded
    
    return resized

def calculate_tracking_metrics(tracks_data: List[List[Dict]]) -> Dict:
    """
    Calculate tracking performance metrics.
    
    Args:
        tracks_data (List[List[Dict]]): Tracking results
        
    Returns:
        Dict: Tracking metrics
    """
    track_ids = set()
    track_switches = 0
    fragmentations = 0
    
    for frame_tracks in tracks_data:
        for track in frame_tracks:
            track_ids.add(track['track_id'])
    
    # Calculate track continuity
    track_presence = {tid: [] for tid in track_ids}
    
    for frame_idx, frame_tracks in enumerate(tracks_data):
        present_ids = {track['track_id'] for track in frame_tracks}
        
        for tid in track_ids:
            track_presence[tid].append(tid in present_ids)
    
    # Count fragmentations (track disappears then reappears)
    for tid, presence in track_presence.items():
        was_present = False
        disappeared = False
        
        for is_present in presence:
            if was_present and not is_present:
                disappeared = True
            elif disappeared and is_present:
                fragmentations += 1
                disappeared = False
            
            was_present = is_present
    
    metrics = {
        'total_tracks': len(track_ids),
        'fragmentations': fragmentations,
        'avg_track_length': np.mean([sum(presence) for presence in track_presence.values()]),
        'track_continuity': 1.0 - (fragmentations / max(len(track_ids), 1))
    }
    
    return metrics

def extract_frames_at_intervals(video_path: str, output_dir: str, 
                               interval_seconds: float = 1.0) -> List[str]:
    """
    Extract frames from video at regular intervals.
    
    Args:
        video_path (str): Input video path
        output_dir (str): Output directory for frames
        interval_seconds (float): Interval between extracted frames
        
    Returns:
        List[str]: List of extracted frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = load_video(video_path)
    props = get_video_properties(cap)
    
    interval_frames = int(props['fps'] * interval_seconds)
    frame_paths = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % interval_frames == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        frame_idx += 1
    
    cap.release()
    return frame_paths 