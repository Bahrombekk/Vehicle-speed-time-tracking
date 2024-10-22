import cv2
import numpy as np
from ultralytics import YOLO
import json

# Load polygon data
with open('polygon.json', 'r') as f:
    polygon_data = json.load(f)

# Extract polygon coordinates
polygon_points = np.array(polygon_data['annotations'][0]['segmentation'][0]).reshape(-1, 2)

def point_in_polygon(point, polygon):
    """Check if point is inside polygon"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Load the YOLO model
model = YOLO("yolov8l.pt")

# Open the video file
video_path = "output-file.mp4"
cap = cv2.VideoCapture(video_path)

# Get original video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object with a different codec
# For Windows, try 'XVID' codec
out = cv2.VideoWriter(
    'car_polygon.avi',  # Change extension to .avi
    cv2.VideoWriter_fourcc(*'XVID'),  # Use XVID codec
    fps,
    (frame_width, frame_height)
)

# Define target classes (2,3,4,5,6,7)
target_classes = [2, 3, 4, 5, 6, 7]

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLO tracking on the frame
        results = model.track(frame, persist=True, classes=target_classes)
        
        if results and results[0].boxes is not None:
            # Get detection boxes
            boxes = results[0].boxes
            
            # Draw polygon
            cv2.polylines(frame, [polygon_points.astype(np.int32)], True, (0, 255, 0), 2)
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Calculate center point of the box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Check if center point is inside polygon
                if point_in_polygon((center_x, center_y), polygon_points):
                    # Draw red box for objects inside polygon
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"Class {class_id}: {conf:.2f}"
                else:
                    # Draw blue box for objects outside polygon
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # Write the frame to output video
        out.write(frame)
        
        # Display the frame
        cv2.imshow("Polygon Detection", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()