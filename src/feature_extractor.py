import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import normalize

class FeatureExtractor:
    """
    Multi-modal feature extractor for player re-identification.
    """
    
    def __init__(self, use_temporal_features: bool = True, 
                 use_motion_features: bool = True,
                 use_color_histogram: bool = False,
                 feature_dim: int = 512):
        """
        Initialize feature extractor.
        
        Args:
            use_temporal_features (bool): Whether to use temporal features
            use_motion_features (bool): Whether to use motion features
            use_color_histogram (bool): Whether to use color histogram features
            feature_dim (int): Dimension of output features
        """
        self.use_temporal_features = use_temporal_features
        self.use_motion_features = use_motion_features
        self.use_color_histogram = use_color_histogram
        self.feature_dim = feature_dim
        
        # Initialize feature extractors
        self.appearance_extractor = self._init_appearance_extractor()
        self.motion_extractor = self._init_motion_extractor() if use_motion_features else None
        
        # Feature history for temporal consistency
        self.feature_history = {}
        self.max_history_length = 30  # Keep last 30 frames of features
    
    def _init_appearance_extractor(self):
        """Initialize appearance feature extractor."""
        # Use a pre-trained CNN for appearance features
        model = torchvision.models.resnet50(pretrained=True)
        # Remove classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model
    
    def _init_motion_extractor(self):
        """Initialize motion feature extractor."""
        # Use optical flow for motion features
        return cv2.optflow.DualTVL1OpticalFlow_create()
    
    def get_feature_dimension(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dim
    
    def extract_combined_features(self, player_crops: List[np.ndarray], 
                                detections: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract combined features from player crops.
        
        Args:
            player_crops (List[np.ndarray]): List of player crop images
            detections (Dict): Detection information
            frame_shape (Tuple[int, int]): Frame dimensions
            
        Returns:
            np.ndarray: Combined features
        """
        if not player_crops:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        
        # Extract appearance features
        appearance_features = self._extract_appearance_features(player_crops)
        
        # Extract motion features if enabled
        motion_features = None
        if self.use_motion_features:
            motion_features = self._extract_motion_features(player_crops)
        
        # Extract temporal features if enabled
        temporal_features = None
        if self.use_temporal_features:
            temporal_features = self._extract_temporal_features(detections, frame_shape)
        
        # Combine features
        features = self._combine_features(
            appearance_features,
            motion_features,
            temporal_features
        )
        
        return features
    
    def _extract_appearance_features(self, player_crops: List[np.ndarray]) -> np.ndarray:
        """Extract appearance features from player crops."""
        features = []
        
        for crop in player_crops:
            # Preprocess image
            img = cv2.resize(crop, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                feat = self.appearance_extractor(img)
                feat = feat.squeeze().numpy()
            
            features.append(feat)
        
        return np.array(features)
    
    def _extract_motion_features(self, player_crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract motion features using optical flow.
        
        Args:
            player_crops (List[np.ndarray]): List of player crop images
            
        Returns:
            np.ndarray: Motion features
        """
        if not player_crops:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        
        # Store previous frame for optical flow
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = None
            self.prev_gray = None
        
        # Resize all crops to a fixed size
        target_size = (64, 64)
        resized_crops = [cv2.resize(crop, target_size) for crop in player_crops]
        
        # Convert current frame to grayscale
        curr_frame = np.stack(resized_crops)
        curr_gray = np.array([cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for crop in resized_crops])
        
        # Initialize motion features
        motion_features = []
        
        if self.prev_gray is not None:
            # Calculate optical flow for each crop
            for i in range(len(resized_crops)):
                prev = self.prev_gray[i] if i < len(self.prev_gray) else np.zeros(target_size, dtype=np.uint8)
                curr = curr_gray[i]
                flow = self.motion_extractor.calc(prev, curr, None)
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hist = cv2.calcHist([magnitude], [0], None, [8], [0, np.max(magnitude) if np.max(magnitude) > 0 else 1])
                hist = hist.flatten() / (np.sum(hist) + 1e-8)
                motion_features.append(hist)
        else:
            # For first frame, use zero motion features
            motion_features = [np.zeros(8) for _ in player_crops]
        
        # Update previous frame
        self.prev_frame = curr_frame
        self.prev_gray = curr_gray
        
        return np.array(motion_features)
    
    def _extract_temporal_features(self, detections: Dict, 
                                 frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract temporal features based on detection history."""
        features = []
        
        for bbox in detections['bboxes']:
            # Normalize bbox coordinates
            x1, y1, x2, y2 = bbox
            norm_bbox = [
                x1 / frame_shape[1],
                y1 / frame_shape[0],
                x2 / frame_shape[1],
                y2 / frame_shape[0]
            ]
            
            # Add temporal context
            temporal_feat = np.array(norm_bbox)
            features.append(temporal_feat)
        
        return np.array(features)
    
    def _combine_features(self, appearance_features: np.ndarray,
                         motion_features: Optional[np.ndarray] = None,
                         temporal_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Combine different feature types into a single feature vector."""
        combined_features = []
        
        for i in range(len(appearance_features)):
            feat = [appearance_features[i]]
            
            if motion_features is not None:
                feat.append(motion_features[i])
            
            if temporal_features is not None:
                feat.append(temporal_features[i])
            
            # Concatenate and normalize
            combined = np.concatenate(feat)
            combined = combined / np.linalg.norm(combined)
            
            # Project to desired dimension if needed
            if len(combined) != self.feature_dim:
                combined = self._project_features(combined)
            
            combined_features.append(combined)
        
        return np.array(combined_features)
    
    def _project_features(self, features: np.ndarray) -> np.ndarray:
        """Project features to desired dimension."""
        if not hasattr(self, '_projection_matrix'):
            # Initialize random projection matrix
            input_dim = len(features)
            self._projection_matrix = np.random.randn(input_dim, self.feature_dim)
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=0)
        
        return np.dot(features, self._projection_matrix)
    
    def update_feature_history(self, track_id: int, features: np.ndarray):
        """Update feature history for a track."""
        if track_id not in self.feature_history:
            self.feature_history[track_id] = []
        
        self.feature_history[track_id].append(features)
        
        # Keep only recent history
        if len(self.feature_history[track_id]) > self.max_history_length:
            self.feature_history[track_id] = self.feature_history[track_id][-self.max_history_length:]
    
    def get_feature_history(self, track_id: int) -> Optional[np.ndarray]:
        """Get feature history for a track."""
        return self.feature_history.get(track_id)
    
    def extract_visual_features(self, player_crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract visual features from player crops using CNN.
        
        Args:
            player_crops (List[np.ndarray]): List of player crop images
            
        Returns:
            np.ndarray: Extracted visual features (n_players x feature_dim)
        """
        if not player_crops:
            return np.array([]).reshape(0, self.feature_dim)
        
        features = []
        
        with torch.no_grad():
            for crop in player_crops:
                if crop.size == 0:
                    # Handle empty crops
                    features.append(np.zeros(self.feature_dim))
                    continue
                
                # Preprocess image
                try:
                    # Convert BGR to RGB
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(crop_rgb).unsqueeze(0)
                    
                    # Extract features
                    feature = self.cnn_features(tensor)
                    feature = feature.squeeze().numpy()
                    features.append(feature)
                    
                except Exception as e:
                    print(f"Error processing crop: {e}")
                    features.append(np.zeros(self.feature_dim))
        
        features = np.array(features)
        
        # Normalize features
        if features.shape[0] > 0:
            features = normalize(features, norm='l2', axis=1)
        
        return features
    
    def extract_color_histogram(self, player_crops: List[np.ndarray], 
                               bins: int = 32) -> np.ndarray:
        """
        Extract color histogram features from player crops.
        
        Args:
            player_crops (List[np.ndarray]): List of player crop images
            bins (int): Number of bins for histogram
            
        Returns:
            np.ndarray: Color histogram features (n_players x bins*3)
        """
        if not player_crops:
            return np.array([]).reshape(0, bins * 3)
        
        histograms = []
        
        for crop in player_crops:
            if crop.size == 0:
                histograms.append(np.zeros(bins * 3))
                continue
            
            # Calculate histogram for each color channel
            hist_b = cv2.calcHist([crop], [0], None, [bins], [0, 256])
            hist_g = cv2.calcHist([crop], [1], None, [bins], [0, 256])
            hist_r = cv2.calcHist([crop], [2], None, [bins], [0, 256])
            
            # Concatenate and normalize
            hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize
            
            histograms.append(hist)
        
        return np.array(histograms)
    
    def extract_spatial_features(self, detections: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract spatial features (position, size) from detections.
        
        Args:
            detections (Dict): Detection results
            frame_shape (Tuple[int, int]): Frame dimensions (height, width)
            
        Returns:
            np.ndarray: Spatial features (n_players x spatial_dim)
        """
        if not detections['bboxes']:
            return np.array([]).reshape(0, 6)
        
        height, width = frame_shape
        spatial_features = []
        
        for i, bbox in enumerate(detections['bboxes']):
            x1, y1, x2, y2 = bbox
            center_x, center_y = detections['centers'][i]
            area = detections['areas'][i]
            
            # Normalize coordinates
            norm_center_x = center_x / width
            norm_center_y = center_y / height
            norm_width = (x2 - x1) / width
            norm_height = (y2 - y1) / height
            norm_area = area / (width * height)
            aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-8)
            
            spatial_feat = [norm_center_x, norm_center_y, norm_width, 
                           norm_height, norm_area, aspect_ratio]
            spatial_features.append(spatial_feat)
        
        return np.array(spatial_features)
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                          metric: str = 'cosine') -> np.ndarray:
        """
        Compute similarity between two sets of features.
        
        Args:
            features1 (np.ndarray): First set of features (n1 x feature_dim)
            features2 (np.ndarray): Second set of features (n2 x feature_dim)
            metric (str): Similarity metric ('cosine', 'euclidean')
            
        Returns:
            np.ndarray: Similarity matrix (n1 x n2)
        """
        if features1.shape[0] == 0 or features2.shape[0] == 0:
            return np.array([]).reshape(features1.shape[0], features2.shape[0])
        
        if metric == 'cosine':
            # Cosine similarity
            similarity = np.dot(features1, features2.T)
        elif metric == 'euclidean':
            # Negative euclidean distance (higher is more similar)
            from scipy.spatial.distance import cdist
            distances = cdist(features1, features2, metric='euclidean')
            similarity = -distances
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def get_feature_dimension(self) -> int:
        """
        Get the total dimension of extracted features.
        
        Returns:
            int: Total feature dimension
        """
        # Calculate total dimension
        visual_dim = self.feature_dim
        spatial_dim = 6  # position, size, aspect ratio
        color_dim = 96 if self.use_color_histogram else 0  # 32 bins * 3 channels
        
        return visual_dim + spatial_dim + color_dim 