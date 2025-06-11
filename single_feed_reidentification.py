#!/usr/bin/env python3
"""
Single Feed Player Re-Identification
Option 2: Re-identification in a Single Feed

This script implements player re-identification for a single 15-second video,
ensuring that players who go out of frame and reappear maintain their original IDs.
"""

import argparse
import os
import sys
import time
from tqdm import tqdm
import cv2
import numpy as np
from typing import Optional, Dict, List
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from player_detector import PlayerDetector
from feature_extractor import FeatureExtractor
from tracker import PlayerTracker
from utils import (load_video, get_video_properties, create_video_writer, 
                  draw_tracks, save_tracking_results, visualize_track_statistics,
                  calculate_tracking_metrics)

class SingleFeedReidentifier:
    """
    Player re-identification system for single video feed.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the re-identification system.
        
        Args:
            model_path (str): Path to YOLOv11 model
            conf_threshold (float): Detection confidence threshold
        """
        print("Initializing Single Feed Re-identification System...")
        
        # Initialize components with improved parameters
        self.detector = PlayerDetector(model_path, conf_threshold)
        self.feature_extractor = FeatureExtractor(
            use_temporal_features=True,  # Enable temporal feature extraction
            use_motion_features=True,    # Enable motion-based features
            feature_dim=512             # Increased feature dimension for better discrimination
        )
        
        # Enhanced tracker parameters
        self.tracker = PlayerTracker(
            max_age=60,            # Increased from 30 to 60 frames to handle longer occlusions
            min_hits=5,            # Increased from 3 to 5 for more stable tracks
            iou_threshold=0.4,     # Increased from 0.3 for stricter association
            feature_threshold=0.6, # Increased from 0.5 for better feature matching
            max_feature_distance=0.8,  # Maximum allowed feature distance
            motion_weight=0.3,     # Weight for motion-based matching
            temporal_weight=0.3,   # Weight for temporal consistency
            appearance_weight=0.4  # Weight for appearance-based matching
        )
        
        # Track management parameters
        self.min_track_length = 10  # Minimum frames for a valid track
        self.max_track_gap = 30     # Maximum frames between track segments
        self.track_history = {}     # Store track history for analysis
        
        print("✓ All components initialized successfully")
    
    def _update_track_history(self, tracks: List[Dict], frame_idx: int):
        """
        Update track history for analysis and consistency.
        
        Args:
            tracks (List[Dict]): Current frame tracks
            frame_idx (int): Current frame index
        """
        for track in tracks:
            track_id = track['track_id']
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'frames': [],
                    'positions': [],
                    'features': [],
                    'last_seen': frame_idx
                }
            
            self.track_history[track_id]['frames'].append(frame_idx)
            self.track_history[track_id]['positions'].append(track['bbox'])
            if 'feature' in track:
                self.track_history[track_id]['features'].append(track['feature'])
            self.track_history[track_id]['last_seen'] = frame_idx
    
    def _filter_tracks(self, tracks: List[Dict], frame_idx: int) -> List[Dict]:
        """
        Filter tracks based on quality and consistency.
        
        Args:
            tracks (List[Dict]): Current frame tracks
            frame_idx (int): Current frame index
            
        Returns:
            List[Dict]: Filtered tracks
        """
        filtered_tracks = []
        
        for track in tracks:
            track_id = track['track_id']
            
            # Skip if track is too short
            if track_id in self.track_history:
                track_length = len(self.track_history[track_id]['frames'])
                if track_length < self.min_track_length:
                    continue
                
                # Check for large gaps in track
                last_seen = self.track_history[track_id]['last_seen']
                if frame_idx - last_seen > self.max_track_gap:
                    continue
            
            filtered_tracks.append(track)
        
        return filtered_tracks
    
    def process_video(self, video_path: str, output_path: str = None, 
                     visualize: bool = True) -> dict:
        """
        Process video for player re-identification.
        
        Args:
            video_path (str): Input video path
            output_path (str, optional): Output video path
            visualize (bool): Whether to create visualization
            
        Returns:
            dict: Processing results and statistics
        """
        print(f"\nProcessing video: {video_path}")
        
        # Load video
        cap = load_video(video_path)
        props = get_video_properties(cap)
        
        print(f"Video properties:")
        print(f"  - Resolution: {props['width']}x{props['height']}")
        print(f"  - FPS: {props['fps']:.2f}")
        print(f"  - Duration: {props['duration']:.2f} seconds")
        print(f"  - Total frames: {props['frame_count']}")
        
        # Set frame shape in tracker
        self.tracker.set_frame_shape((props['height'], props['width']))
        
        # Initialize video writer if needed
        writer = None
        if output_path and visualize:
            writer = create_video_writer(
                output_path, props['fps'], 
                props['width'], props['height']
            )
            print(f"Output video will be saved to: {output_path}")
        
        # Process frames
        all_tracks = []
        frame_idx = 0
        start_time = time.time()
        
        print("\nProcessing frames...")
        pbar = tqdm(total=props['frame_count'], desc="Tracking players")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players
            detections = self.detector.detect_frame(frame)
            print(f"\nFrame {frame_idx}:")
            print(f"  Raw detections: {len(detections['bboxes'])} players")
            
            detections = self.detector.filter_player_detections(detections)
            print(f"  After filtering: {len(detections['bboxes'])} players")
            
            if detections['bboxes']:
                # Extract features
                player_crops = self.detector.extract_player_crops(frame, detections)
                print(f"  Extracted {len(player_crops)} player crops")
                
                features = self.feature_extractor.extract_combined_features(
                    player_crops, detections, (props['height'], props['width'])
                )
                print(f"  Extracted features shape: {features.shape}")
                
                # Update tracker
                tracks = self.tracker.update(detections, features)
                
                # Update track history and filter tracks
                self._update_track_history(tracks, frame_idx)
                tracks = self._filter_tracks(tracks, frame_idx)
                
                print(f"  Active tracks: {len(tracks)}")
            else:
                # No detections - update tracker with empty detections
                feature_dim = self.feature_extractor.get_feature_dimension()
                empty_features = np.empty((0, feature_dim), dtype=np.float32)
                tracks = self.tracker.update({'bboxes': [], 'confidences': []}, empty_features)
                print("  No detections in frame")
            
            all_tracks.append(tracks)
            
            # Visualize if needed
            if visualize and writer:
                viz_frame = draw_tracks(frame, tracks)
                writer.write(viz_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        metrics = calculate_tracking_metrics(all_tracks)
        
        # Add track history statistics
        metrics.update({
            'avg_track_duration': np.mean([
                len(history['frames']) 
                for history in self.track_history.values()
            ]),
            'max_track_duration': max([
                len(history['frames']) 
                for history in self.track_history.values()
            ]) if self.track_history else 0,
            'track_consistency': self._calculate_track_consistency()
        })
        
        results = {
            'video_path': video_path,
            'output_path': output_path,
            'processing_time': processing_time,
            'fps_processed': props['frame_count'] / processing_time,
            'video_properties': props,
            'tracking_metrics': metrics,
            'total_frames_processed': frame_idx,
            'tracks_data': all_tracks
        }
        
        print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
        print(f"  - Processing FPS: {results['fps_processed']:.2f}")
        print(f"  - Total unique tracks: {metrics['total_tracks']}")
        print(f"  - Average track length: {metrics['avg_track_length']:.1f} frames")
        print(f"  - Track continuity: {metrics['track_continuity']:.2f}")
        print(f"  - Track consistency: {metrics['track_consistency']:.2f}")
        
        return results
    
    def _calculate_track_consistency(self) -> float:
        """
        Calculate track consistency score based on track history.
        
        Returns:
            float: Track consistency score (0-1)
        """
        if not self.track_history:
            return 0.0
        
        consistency_scores = []
        
        for track_id, history in self.track_history.items():
            if len(history['frames']) < 2:
                continue
            
            # Calculate position consistency
            positions = np.array(history['positions'])
            position_changes = np.diff(positions, axis=0)
            position_consistency = 1.0 / (1.0 + np.mean(np.linalg.norm(position_changes, axis=1)))
            
            # Calculate feature consistency if available
            feature_consistency = 1.0
            if len(history['features']) > 1:
                features = np.array(history['features'])
                feature_changes = np.diff(features, axis=0)
                feature_consistency = 1.0 / (1.0 + np.mean(np.linalg.norm(feature_changes, axis=1)))
            
            # Combine scores
            consistency_scores.append(0.7 * position_consistency + 0.3 * feature_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def save_results(self, results: dict, output_dir: str):
        """
        Save processing results and visualizations.
        
        Args:
            results (dict): Processing results
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tracking results
        tracks_path = os.path.join(output_dir, 'tracking_results.json')
        save_tracking_results(results['tracks_data'], tracks_path)
        
        # Generate statistics visualizations
        print("Generating tracking statistics...")
        stats_dir = os.path.join(output_dir, 'statistics')
        visualize_track_statistics(results['tracks_data'], stats_dir)
        
        # Save summary
        summary = {k: v for k, v in results.items() if k != 'tracks_data'}
        summary_path = os.path.join(output_dir, 'processing_summary.json')
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {output_dir}")

def process_video(input_path: str, model_path: str, output_path: Optional[str] = None,
                 output_dir: Optional[str] = None, conf_threshold: float = 0.5,
                 batch_size: int = 4, frame_skip: int = 1, visualize: bool = True) -> Dict:
    """
    Process a video file for player re-identification.
    
    Args:
        input_path (str): Path to input video
        model_path (str): Path to YOLO model
        output_path (Optional[str]): Path to output video
        output_dir (Optional[str]): Directory for output files
        conf_threshold (float): Detection confidence threshold
        batch_size (int): Number of frames to process in batch
        frame_skip (int): Number of frames to skip between processing
        visualize (bool): Whether to create visualization
        
    Returns:
        Dict: Processing summary
    """
    # Initialize components
    detector = PlayerDetector(model_path, conf_threshold=conf_threshold)
    feature_extractor = FeatureExtractor(use_temporal_features=True)
    tracker = PlayerTracker(
        max_age=60,            # Increased from 30 to 60 frames to handle longer occlusions
        min_hits=5,            # Increased from 3 to 5 for more stable tracks
        iou_threshold=0.4,     # Increased from 0.3 for stricter association
        feature_threshold=0.6, # Increased from 0.5 for better feature matching
        max_feature_distance=0.8,  # Maximum allowed feature distance
        motion_weight=0.3,     # Weight for motion-based matching
        temporal_weight=0.3,   # Weight for temporal consistency
        appearance_weight=0.4,  # Weight for appearance-based matching
        frame_shape=(video_props['height'], video_props['width'])  # Pass frame shape from video properties
    )
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    video_props = get_video_properties(cap)
    
    # Initialize video writer if needed
    writer = None
    if visualize and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, video_props['fps'],
                               (video_props['width'], video_props['height']))
    
    # Initialize tracking results
    tracking_results = []
    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    
    # Process frames in batches
    while True:
        batch_frames = []
        batch_indices = []
        
        # Read batch of frames
        for _ in range(batch_size):
            for _ in range(frame_skip):
                ret = cap.grab()
                if not ret:
                    break
                frame_count += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            
            batch_frames.append(frame)
            batch_indices.append(frame_count)
            frame_count += 1
        
        if not batch_frames:
            break
        
        # Process batch
        batch_detections = []
        batch_features = []
        
        for frame in batch_frames:
            # Detect players
            detections = detector.detect_frame(frame)
            detections = detector.filter_player_detections(detections)
            
            # Extract player crops
            player_crops = detector.extract_player_crops(frame, detections)
            
            # Extract features
            features = feature_extractor.extract_combined_features(
                player_crops, detections, frame.shape[:2]
            )
            
            batch_detections.append(detections)
            batch_features.append(features)
        
        # Update tracker for each frame in batch
        for i, (detections, features) in enumerate(zip(batch_detections, batch_features)):
            # Update tracker
            tracks = tracker.update(detections, features)
            tracking_results.append(tracks)
            
            # Visualize if needed
            if visualize:
                frame = batch_frames[i]
                frame = draw_tracks(frame, tracks)
                
                if writer:
                    writer.write(frame)
            
            processed_frames += 1
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    
    # Calculate processing time
    processing_time = time.time() - start_time
    fps_processed = processed_frames / processing_time
    
    # Save tracking results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tracking results
        tracking_path = os.path.join(output_dir, 'tracking_results.json')
        save_tracking_results(tracking_results, tracking_path)
        
        # Generate and save visualizations
        visualize_track_statistics(tracking_results, output_dir)
        
        # Calculate and save metrics
        metrics = calculate_tracking_metrics(tracking_results)
        metrics_path = os.path.join(output_dir, 'tracking_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Create processing summary
    summary = {
        'video_path': input_path,
        'output_path': output_path,
        'processing_time': processing_time,
        'fps_processed': fps_processed,
        'video_properties': video_props,
        'tracking_metrics': metrics if output_dir else {},
        'total_frames_processed': processed_frames
    }
    
    # Save summary
    if output_dir:
        summary_path = os.path.join(output_dir, 'processing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print("Processing completed. Saving results...")
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Single Feed Player Re-Identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/15sec_input_720p.mp4',
        help='Input video path'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='best.pt',
        help='Path to YOLOv11 model'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/single_feed_output.mp4',
        help='Output video path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/single_feed/',
        help='Output directory for all results'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization (faster processing)'
    )
    
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Number of frames to skip between processing'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize system
        reidentifier = SingleFeedReidentifier(
            model_path=args.model,
            conf_threshold=args.conf_threshold
        )
        
        # Process video
        results = reidentifier.process_video(
            video_path=args.input,
            output_path=args.output if not args.no_visualize else None,
            visualize=not args.no_visualize
        )
        
        # Save results
        reidentifier.save_results(results, args.output_dir)
        
        print("\n" + "="*60)
        print("SINGLE FEED RE-IDENTIFICATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Input video: {args.input}")
        print(f"Output video: {args.output}")
        print(f"Results directory: {args.output_dir}")
        print(f"Total processing time: {results['processing_time']:.2f} seconds")
        print(f"Unique players tracked: {results['tracking_metrics']['total_tracks']}")
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 