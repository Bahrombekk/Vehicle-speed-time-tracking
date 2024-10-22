import cv2
import numpy as np
from ultralytics import YOLO
import json
from time import time
from collections import defaultdict

# Load polygon data
with open('polygon.json', 'r') as f:
    polygon_data = json.load(f)

# Extract polygon coordinates
polygon_points = np.array(polygon_data['annotations'][0]['segmentation'][0]).reshape(-1, 2)

# Define polygon length in meters
POLYGON_LENGTH = 8  # meters

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

def calculate_speed(distance_meters, time_seconds):
    """Calculate speed in km/h given distance in meters and time in seconds"""
    if time_seconds == 0:
        return 0
    speed_mps = distance_meters / time_seconds
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def calculate_current_speed(distance_meters, current_time, start_time):
    """Calculate current speed based on partial time"""
    time_diff = current_time - start_time
    if time_diff > 0:
        return calculate_speed(distance_meters, time_diff)
    return 0

# Load the YOLO model
model = YOLO("model/yolov8l.pt")

# Open the video file
video_path = "video/output-file.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter
out = cv2.VideoWriter(
    'vehicle_speed_time_tracking.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# Define target classes
target_classes = [2, 3, 4, 5, 6, 7]

# Dictionary to store vehicle tracking data
vehicle_tracking = defaultdict(lambda: {
    'start_time': None,
    'end_time': None,
    'in_polygon': False,
    'total_time': 0,
    'speed': 0,
    'current_speed': 0,
    'class_id': None
})

# Class names mapping
class_names = {
    2: 'Car',
    3: 'Motorcycle',
    4: 'Bus',
    5: 'Train',
    6: 'Truck',
    7: 'Boat'
}

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        current_time = frame_count / fps
        frame_count += 1
        
        results = model.track(frame, persist=True, classes=target_classes)
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Draw polygon
            cv2.polylines(frame, [polygon_points.astype(np.int32)], True, (0, 255, 0), 2)
            
            current_ids = set()
            
            for box in boxes:
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                    current_ids.add(track_id)
                    
                    class_id = int(box.cls[0])
                    vehicle_tracking[track_id]['class_id'] = class_id
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    is_inside = point_in_polygon((center_x, center_y), polygon_points)
                    
                    if is_inside:
                        if not vehicle_tracking[track_id]['in_polygon']:
                            vehicle_tracking[track_id]['start_time'] = current_time
                            vehicle_tracking[track_id]['in_polygon'] = True
                        
                        # Calculate current speed while in polygon
                        current_duration = current_time - vehicle_tracking[track_id]['start_time']
                        current_speed = calculate_current_speed(POLYGON_LENGTH, current_time, 
                                                             vehicle_tracking[track_id]['start_time'])
                        vehicle_tracking[track_id]['current_speed'] = current_speed
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    else:
                        if vehicle_tracking[track_id]['in_polygon']:
                            vehicle_tracking[track_id]['end_time'] = current_time
                            vehicle_tracking[track_id]['in_polygon'] = False
                            total_time = vehicle_tracking[track_id]['end_time'] - vehicle_tracking[track_id]['start_time']
                            vehicle_tracking[track_id]['total_time'] = total_time
                            speed = calculate_speed(POLYGON_LENGTH, total_time)
                            vehicle_tracking[track_id]['speed'] = speed
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    # Display both speed and time information
                    if vehicle_tracking[track_id]['in_polygon']:
                        current_duration = current_time - vehicle_tracking[track_id]['start_time']
                        current_speed = vehicle_tracking[track_id]['current_speed']
                        info_text = f"ID:{track_id} {class_names[class_id]}"
                        cv2.putText(frame, info_text, (int(x1), int(y1)-20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (15, 226, 250), 1)
                        time_text = f"Time: {current_duration:.1f}s"
                        cv2.putText(frame, time_text, (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (15, 226, 250), 1)
                        speed_text = f"Speed: {current_speed:.1f}km/h"
                        cv2.putText(frame, speed_text, (int(x1), int(y1)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (15, 226, 250), 1)
                    elif vehicle_tracking[track_id]['total_time'] > 0:
                        info_text = f"ID:{track_id} {class_names[class_id]}"
                        cv2.putText(frame, info_text, (int(x1), int(y1)-20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (15, 226, 250), 1)
                        time_text = f"Total: {vehicle_tracking[track_id]['total_time']:.1f}s"
                        cv2.putText(frame, time_text, (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (15, 226, 250), 1)
                        speed_text = f"Avg: {vehicle_tracking[track_id]['speed']:.1f}km/h"
                        cv2.putText(frame, speed_text, (int(x1), int(y1)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (15, 226, 250), 1)
        
        out.write(frame)
        cv2.imshow("Vehicle Speed and Time Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Print final results
print("\nFinal Vehicle Statistics:")
print("ID | Type | Time(s) | Speed(km/h)")
print("-" * 40)
for vehicle_id, data in vehicle_tracking.items():
    if data['total_time'] > 0:
        vehicle_type = class_names[data['class_id']]
        print(f"{vehicle_id:2d} | {vehicle_type:6s} | {data['total_time']:6.2f}s | {data['speed']:6.1f}km/h")

cap.release()
out.release()
cv2.destroyAllWindows()