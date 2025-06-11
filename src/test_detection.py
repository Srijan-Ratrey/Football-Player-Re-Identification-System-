import cv2
from player_detector import PlayerDetector

VIDEO_PATH = '/Users/srijanratrey/Documents/Learning and coding/Football Analysis/data/15sec_input_720p.mp4'
MODEL_PATH = '/Users/srijanratrey/Documents/Learning and coding/Football Analysis/yolov8n.pt'  # or use 'best.pt' if that's your custom model

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Could not open video: {VIDEO_PATH}")
    exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read the first frame.")
    exit(1)

# Initialize detector
print("Loading model...")
detector = PlayerDetector(MODEL_PATH, conf_threshold=0.5)

# Run detection
print("Running detection on the first frame...")
detections = detector.detect_frame(frame)

print("Detection results:")
print(detections)

# Optionally, visualize detections
viz = detector.visualize_detections(frame, detections)
cv2.imwrite("/Users/srijanratrey/Documents/Learning and coding/Football Analysis/results/test_detection_output.jpg", viz)
print("Visualization saved to /Users/srijanratrey/Documents/Learning and coding/Football Analysis/results/test_detection_output.jpg") 