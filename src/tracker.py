import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import cv2

class Track:
    """
    Individual track object for a single player.
    """
    
    def __init__(self, track_id: int, bbox: List[float], feature: np.ndarray, frame_id: int, max_age: int, min_hits: int, frame_shape: Tuple[int, int]):
        """
        Initialize a track.
        
        Args:
            track_id (int): Unique track ID
            bbox (List[float]): Bounding box [x1, y1, x2, y2]
            feature (np.ndarray): Feature vector for the track
            frame_id (int): Frame number where track was initialized
            max_age (int): Maximum age for the track
            min_hits (int): Minimum hits for confirmation
            frame_shape (Tuple[int, int]): Frame shape (height, width)
        """
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.last_frame = frame_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_shape = frame_shape
        
        # Initialize Kalman filter for position prediction
        self.kf = self._init_kalman_filter(bbox)
        
        # Store track history
        self.bbox_history = [bbox]
        self.feature_history = [feature]
        self.frame_history = [frame_id]
        
    def _init_kalman_filter(self, bbox: List[float]):
        """
        Initialize Kalman filter for tracking.
        
        Args:
            bbox (List[float]): Initial bounding box
            
        Returns:
            cv2.KalmanFilter: Initialized Kalman filter
        """
        kf = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurements
        
        # State: [x_center, y_center, width, height, dx, dy, dw, dh]
        # Measurement: [x_center, y_center, width, height]
        
        # Transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Error covariance
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        # Initialize state
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        kf.statePre = np.array([x_center, y_center, width, height, 0, 0, 0, 0], 
                              dtype=np.float32)
        kf.statePost = kf.statePre.copy()
        
        return kf
    
    def predict(self):
        """
        Predict the next state using Kalman filter.
        
        Returns:
            List[float]: Predicted bounding box [x1, y1, x2, y2]
        """
        pred_state = self.kf.predict()
        
        x_center, y_center, width, height = pred_state[:4]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return [x1, y1, x2, y2]
    
    def update(self, bbox: List[float], feature: np.ndarray, frame_id: int, confidence: float = 1.0):
        """
        Update track with new detection.
        
        Args:
            bbox (List[float]): New bounding box
            feature (np.ndarray): New feature vector
            frame_id (int): Current frame number
            confidence (float): Detection confidence score
        """
        self.bbox = bbox
        self.feature = feature
        self.last_frame = frame_id
        self.hits += 1
        self.time_since_update = 0
        
        # Update Kalman filter
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        measurement = np.array([x_center, y_center, width, height], dtype=np.float32)
        self.kf.correct(measurement)
        
        # Update history
        self.bbox_history.append(bbox)
        self.feature_history.append(feature)
        self.frame_history.append(frame_id)
        
        # Keep only recent history
        max_history = 30
        if len(self.bbox_history) > max_history:
            self.bbox_history = self.bbox_history[-max_history:]
            self.feature_history = self.feature_history[-max_history:]
            self.frame_history = self.frame_history[-max_history:]
    
    def increment_age(self):
        """Increment age and time since update."""
        self.age += 1
        self.time_since_update += 1
    
    def is_active(self) -> bool:
        """
        Check if track is still active.
        
        Returns:
            bool: True if track is active, False otherwise
        """
        return self.time_since_update < self.max_age
    
    def is_confirmed(self) -> bool:
        """
        Check if track is confirmed (has enough hits).
        
        Returns:
            bool: True if track is confirmed, False otherwise
        """
        return self.hits >= self.min_hits
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.increment_age()
        self.time_since_update += 1
    
    def get_average_feature(self, n_recent: int = 5) -> np.ndarray:
        """
        Get average feature from recent detections.
        
        Args:
            n_recent (int): Number of recent features to average
            
        Returns:
            np.ndarray: Average feature vector
        """
        if not self.feature_history:
            return self.feature
        
        recent_features = self.feature_history[-n_recent:]
        return np.mean(recent_features, axis=0)

    def get_state(self) -> List[float]:
        """
        Get the current state of the track.
        
        Returns:
            List[float]: Current bounding box [x1, y1, x2, y2]
        """
        return self.bbox

    def get_feature(self) -> np.ndarray:
        """
        Get the current feature vector of the track.
        
        Returns:
            np.ndarray: Current feature vector
        """
        return self.feature

    def get_predicted_state(self) -> List[float]:
        """
        Get the predicted state of the track from the Kalman filter.
        
        Returns:
            List[float]: Predicted bounding box [x1, y1, x2, y2]
        """
        pred_state = self.kf.predict()
        x_center, y_center, width, height = pred_state[:4]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]

    def to_dict(self) -> Dict:
        """
        Convert track to dictionary format.
        
        Returns:
            Dict: Track information
        """
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'feature': self.feature.tolist() if isinstance(self.feature, np.ndarray) else self.feature,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'is_confirmed': self.is_confirmed()
        }


class PlayerTracker:
    """
    Multi-object tracker using Kalman filter and feature matching.
    """
    
    def __init__(self, max_age: int = 60, min_hits: int = 5,
                 iou_threshold: float = 0.4, feature_threshold: float = 0.6,
                 max_feature_distance: float = 0.8, motion_weight: float = 0.3,
                 temporal_weight: float = 0.3, appearance_weight: float = 0.4,
                 frame_shape: Tuple[int, int] = (720, 1280)):
        """
        Initialize tracker.
        
        Args:
            max_age (int): Maximum number of frames to keep track without updates
            min_hits (int): Minimum number of detections to confirm track
            iou_threshold (float): IoU threshold for association
            feature_threshold (float): Feature similarity threshold
            max_feature_distance (float): Maximum allowed feature distance
            motion_weight (float): Weight for motion-based matching
            temporal_weight (float): Weight for temporal consistency
            appearance_weight (float): Weight for appearance-based matching
            frame_shape (Tuple[int, int]): Frame shape (height, width)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.max_feature_distance = max_feature_distance
        self.frame_shape = frame_shape
        
        # Feature matching weights
        self.motion_weight = motion_weight
        self.temporal_weight = temporal_weight
        self.appearance_weight = appearance_weight
        
        # Track management
        self.tracks = []
        self.frame_count = 0
        self.next_id = 0
        
        # Kalman filter parameters
        self.kf_params = {
            'dt': 1.0,  # Time step
            'process_noise_scale': 0.1,
            'measurement_noise_scale': 0.1
        }
    
    def update(self, detections: Dict, features: np.ndarray) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections (Dict): Detection results
            features (np.ndarray): Detection features
            
        Returns:
            List[Dict]: Updated tracks
        """
        self.frame_count += 1
        
        # Get detection boxes and features
        bboxes = detections.get('bboxes', [])
        confidences = detections.get('confidences', [])
        
        # Predict new locations of existing tracks
        self._predict()
        
        # Associate detections to existing tracks
        matches, unmatched_detections, unmatched_tracks = self._associate(
            bboxes, features
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                bbox=bboxes[det_idx],
                feature=features[det_idx],
                frame_id=self.frame_count,
                confidence=confidences[det_idx]
            )
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(
                bboxes[det_idx],
                features[det_idx],
                confidences[det_idx]
            )
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.is_active()]
        
        # Return active tracks
        return [t.to_dict() for t in self.tracks if t.is_confirmed()]
    
    def _predict(self):
        """Predict new locations of existing tracks."""
        for track in self.tracks:
            track.predict()
    
    def _associate(self, bboxes: List[np.ndarray], 
                  features: np.ndarray) -> Tuple[List[Tuple[int, int]], 
                                               List[int], List[int]]:
        """
        Associate detections to existing tracks.
        
        Args:
            bboxes (List[np.ndarray]): Detection boxes
            features (np.ndarray): Detection features
            
        Returns:
            Tuple[List[Tuple[int, int]], List[int], List[int]]: 
                Matches, unmatched detections, unmatched tracks
        """
        if not self.tracks:
            return [], list(range(len(bboxes))), []
        
        if not bboxes:
            return [], [], list(range(len(self.tracks)))
        
        # Calculate cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(bboxes)))
        
        for i, track in enumerate(self.tracks):
            for j, (bbox, feature) in enumerate(zip(bboxes, features)):
                # Calculate IoU cost
                iou_cost = 1.0 - self._calculate_iou(track.get_state(), bbox)
                
                # Calculate feature cost
                feature_cost = 1.0 - self._calculate_feature_similarity(
                    track.get_feature(), feature
                )
                
                # Calculate motion cost
                motion_cost = self._calculate_motion_cost(track, bbox)
                
                # Calculate temporal cost
                temporal_cost = self._calculate_temporal_cost(track)
                
                # Combine costs with weights
                cost = (
                    self.appearance_weight * feature_cost +
                    self.motion_weight * motion_cost +
                    self.temporal_weight * temporal_cost +
                    (1.0 - self.appearance_weight - self.motion_weight - 
                     self.temporal_weight) * iou_cost
                )
                
                cost_matrix[i, j] = cost
        
        # Perform Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on cost threshold
        matches = []
        unmatched_detections = list(range(len(bboxes)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > self.max_feature_distance:
                unmatched_detections.append(j)
                unmatched_tracks.append(i)
            else:
                matches.append((i, j))
                if j in unmatched_detections:
                    unmatched_detections.remove(j)
                if i in unmatched_tracks:
                    unmatched_tracks.remove(i)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _create_track(self, bbox: np.ndarray, feature: np.ndarray, 
                     confidence: float):
        """Create new track."""
        track = Track(
            self.next_id,
            bbox,
            feature,
            self.frame_count,
            self.max_age,
            self.min_hits,
            self.frame_shape
        )
        self.tracks.append(track)
        self.next_id += 1
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / (area1 + area2 - intersection)
    
    def _calculate_feature_similarity(self, feat1: np.ndarray, 
                                    feat2: np.ndarray) -> float:
        """Calculate cosine similarity between features."""
        return np.dot(feat1, feat2) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2)
        )
    
    def _calculate_motion_cost(self, track: 'Track', bbox: np.ndarray) -> float:
        """Calculate motion-based cost."""
        predicted_state = track.get_predicted_state()
        current_state = bbox
        
        # Calculate center points
        pred_center = np.array([
            (predicted_state[0] + predicted_state[2]) / 2,
            (predicted_state[1] + predicted_state[3]) / 2
        ])
        curr_center = np.array([
            (current_state[0] + current_state[2]) / 2,
            (current_state[1] + current_state[3]) / 2
        ])
        
        # Calculate distance
        distance = np.linalg.norm(pred_center - curr_center)
        
        # Normalize distance
        max_distance = np.sqrt(
            track.frame_shape[0]**2 + track.frame_shape[1]**2
        )
        return distance / max_distance
    
    def _calculate_temporal_cost(self, track: 'Track') -> float:
        """Calculate temporal consistency cost."""
        if track.time_since_update > self.max_age:
            return 1.0
        
        return track.time_since_update / self.max_age

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID.
        
        Args:
            track_id (int): Track ID to search for
            
        Returns:
            Optional[Track]: Track object if found, None otherwise
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = []
        self.frame_count = 0
        self.next_id = 0

    def set_frame_shape(self, frame_shape: Tuple[int, int]):
        """
        Update frame shape for all tracks.
        
        Args:
            frame_shape (Tuple[int, int]): New frame shape (height, width)
        """
        self.frame_shape = frame_shape
        for track in self.tracks:
            track.frame_shape = frame_shape 