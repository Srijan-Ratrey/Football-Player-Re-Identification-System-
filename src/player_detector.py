import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict

class PlayerDetector:
    """
    Player detector using YOLOv11 model for detecting players and ball in football footage.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = 'auto'):
        """
        Initialize the player detector.
        
        Args:
            model_path (str): Path to the YOLOv11 model file
            conf_threshold (float): Confidence threshold for detections
            device (str): Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Class names (assuming standard YOLO format)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        
    def detect_frame(self, frame: np.ndarray) -> Dict:
        """
        Detect players and ball in a single frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            
        Returns:
            Dict: Detection results containing bboxes, confidences, class_ids, and features
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, device=self.device)
        
        detections = {
            'bboxes': [],
            'confidences': [],
            'class_ids': [],
            'centers': [],
            'areas': []
        }
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                # Extract detection information
                xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                conf = boxes.conf.cpu().numpy()  # Confidence scores
                cls = boxes.cls.cpu().numpy()    # Class IDs
                
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    confidence = conf[i]
                    class_id = int(cls[i])
                    
                    # Calculate center and area
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections['bboxes'].append([x1, y1, x2, y2])
                    detections['confidences'].append(confidence)
                    detections['class_ids'].append(class_id)
                    detections['centers'].append([center_x, center_y])
                    detections['areas'].append(area)
        
        return detections
    
    def extract_player_crops(self, frame: np.ndarray, detections: Dict) -> List[np.ndarray]:
        """
        Extract cropped regions of detected players.
        
        Args:
            frame (np.ndarray): Input frame
            detections (Dict): Detection results
            
        Returns:
            List[np.ndarray]: List of cropped player images
        """
        crops = []
        
        for bbox in detections['bboxes']:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding around bounding box
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
            else:
                # If crop is empty, append a small placeholder
                crops.append(np.zeros((32, 32, 3), dtype=np.uint8))
                
        return crops
    
    def filter_player_detections(self, detections: Dict, min_area: int = 500) -> Dict:
        """
        Filter detections to keep only players (remove ball and other objects).
        
        Args:
            detections (Dict): Detection results
            min_area (int): Minimum area threshold for player detections
            
        Returns:
            Dict: Filtered detections containing only players
        """
        filtered_detections = {
            'bboxes': [],
            'confidences': [],
            'class_ids': [],
            'centers': [],
            'areas': []
        }
        
        print(f"\nFiltering detections:")
        print(f"  Total detections before filtering: {len(detections['bboxes'])}")
        print(f"  Minimum area threshold: {min_area}")
        
        for i in range(len(detections['bboxes'])):
            area = detections['areas'][i]
            class_id = detections['class_ids'][i]
            confidence = detections['confidences'][i]
            
            # Filter based on area and class (accepting class 2 as players)
            if area >= min_area and class_id in [0, 2]:  # Accept both class 0 and 2 as players
                filtered_detections['bboxes'].append(detections['bboxes'][i])
                filtered_detections['confidences'].append(detections['confidences'][i])
                filtered_detections['class_ids'].append(detections['class_ids'][i])
                filtered_detections['centers'].append(detections['centers'][i])
                filtered_detections['areas'].append(detections['areas'][i])
            else:
                print(f"  Filtered out detection {i}:")
                print(f"    - Area: {area:.1f} (threshold: {min_area})")
                print(f"    - Class ID: {class_id} (expected: 0 or 2)")
                print(f"    - Confidence: {confidence:.3f}")
        
        print(f"  Detections after filtering: {len(filtered_detections['bboxes'])}")
        
        return filtered_detections
    
    def visualize_detections(self, frame: np.ndarray, detections: Dict, 
                           player_ids: List[int] = None) -> np.ndarray:
        """
        Visualize detections on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (Dict): Detection results
            player_ids (List[int], optional): Player IDs for tracking visualization
            
        Returns:
            np.ndarray: Frame with visualized detections
        """
        viz_frame = frame.copy()
        
        for i, bbox in enumerate(detections['bboxes']):
            x1, y1, x2, y2 = map(int, bbox)
            confidence = detections['confidences'][i]
            
            # Draw bounding box
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Player {player_ids[i] if player_ids else i}: {confidence:.2f}"
            cv2.putText(viz_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return viz_frame
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'class_names': self.class_names
        } 