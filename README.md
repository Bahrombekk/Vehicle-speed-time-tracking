Here's a well-structured README for your project:

---

# Vehicle Speed and Time Tracking with YOLO and OpenCV

This project uses YOLO (You Only Look Once) object detection and OpenCV to track vehicles in a video, measure their speed, and display the time they spend within a predefined polygon. The system identifies multiple vehicle classes, calculates their speeds in real time, and logs the time spent in a specific area.

## Features
- **Real-Time Vehicle Tracking**: Detects vehicles like cars, buses, and trucks using YOLO.
- **Speed Calculation**: Measures the speed of each vehicle in km/h based on a defined polygon length.
- **Time Tracking**: Records the time vehicles stay within the polygon.
- **Visualization**: Draws bounding boxes around vehicles and displays speed and time information on the video.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLO
- Pretrained YOLO model (`yolov8l.pt`)
- Video file for processing

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your video file in the `video/` folder and the YOLO model in the `model/` folder.

## Usage
Run the script using:
```bash
python vehicle_tracking.py
```

## Output
- **Tracked Vehicles**: The video output highlights detected vehicles with their speeds and time spent in the area.
- **Final Statistics**: Prints a summary of vehicle statistics, including type, time, and speed, to the terminal.

## Configuration
- **Polygon**: The polygon area can be adjusted by editing `polygon.json`. Ensure the coordinates represent the region of interest.
- **Vehicle Classes**: Modify the `target_classes` list to add or remove vehicle types from detection.

## Example
Here's an example output:
```
Final Vehicle Statistics:
ID | Type    | Time(s) | Speed(km/h)
------------------------------------
1  | Car     | 5.30s   | 45.2km/h
2  | Truck   | 3.10s   | 35.8km/h
```

## License
This project is licensed under the MIT License.

---

This README outlines the project's purpose, usage, and configuration clearly for potential users or developers.
