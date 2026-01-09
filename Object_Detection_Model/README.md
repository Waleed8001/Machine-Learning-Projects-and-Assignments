
## 📱 Real-Time Object Detection (Android + YOLO Python)

This project demonstrates **real-time object detection** using an **Android application (CameraX + Kotlin)** connected to a **Python YOLO server** via **TCP sockets**.

The Android app captures live camera frames, sends them to a Python server where **YOLO** performs object detection, and receives **JSON results** to draw bounding boxes and labels on detected objects.


## 🚀 Features
- Live camera feed using **CameraX**
- Real-time object detection using **YOLO**
- Android ↔ Python communication via **TCP Socket**
- Bounding boxes with class names & confidence
- Lightweight Android processing
- JSON-based detection results
- Real-time performance


## 🛠️ Tech Stack

### 📱 Android App
- Kotlin
- CameraX
- TCP Socket
- Bitmap & Canvas
- JSON Parsing

### 🐍 Python Server
- Python 3
- YOLO (Ultralytics)
- OpenCV
- NumPy
- Socket Programming


## 📐 System Architecture

Android Camera ↓ ImageProxy (CameraX) ↓ Bitmap → JPEG ↓ TCP Socket ↓ Python YOLO Server ↓ JSON (class, confidence, box) ↓ Android App ↓ Draw Bounding Boxes ↓ Live Display

## 📱 Android App Workflow
1. Capture camera frames using **CameraX**
2. Convert `ImageProxy` to `Bitmap`
3. Compress Bitmap to **JPEG**
4. Send image bytes to Python server
5. Receive JSON detection results
6. Parse JSON
7. Draw bounding boxes using **Canvas**
8. Display output in `ImageView`



## 🐍 Python Server Workflow
1. Listen for Android connection
2. Receive image size (4 bytes)
3. Receive JPEG image bytes
4. Decode image using OpenCV
5. Run YOLO object detection
6. Convert detections to JSON
7. Send JSON back to Android

## 📂 Project Structure

YOLO-Android-Object-Detection/ │ ├── android-app/ │   ├── MainActivity.kt │   ├── activity_main.xml │   └── AndroidManifest.xml │ ├── python-server/ │   ├── server.py │   └── requirements.txt │ └── README.md

## 🐍 Python Server Setup

```bash
### 1️⃣ Install Dependencies

pip install ultralytics opencv-python numpy

2️⃣ Python YOLO Server (server.py)

import socket
import struct
import json
import cv2
import numpy as np
from ultralytics import YOLO

HOST = "0.0.0.0"
PORT = 9999

model = YOLO("yolov8n.pt")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("✅ YOLO Server Started")

conn, addr = server.accept()
print("📱 Android Connected:", addr)

while True:
    try:
        # Receive image size
        size_data = conn.recv(4)
        if not size_data:
            break

        img_size = struct.unpack(">I", size_data)[0]

        # Receive image bytes
        img_bytes = b""
        while len(img_bytes) < img_size:
            img_bytes += conn.recv(img_size - len(img_bytes))

        # Decode image
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Run YOLO
        results = model(frame)[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

        # Send JSON back
        json_data = json.dumps(detections).encode("utf-8")
        conn.sendall(struct.pack(">I", len(json_data)))
        conn.sendall(json_data)

    except Exception as e:
        print("❌ Error:", e)
        break

conn.close()
server.close()

```
📱 Android Permissions

```bash
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.INTERNET"/>
```
🎯 Use Cases

Person detection

Vehicle detection

Security surveillance

Smart monitoring systems

Educational AI applications

🔮 Future Improvements

Object tracking

On-device YOLO (TensorFlow Lite)

FPS optimization

Detection filtering

Multiple device support

🧠 Key Concept

> The Android device handles image capture and display, while heavy object detection is performed on the Python server for better performance.

👨‍💻 Author

Owais Ahmed | Waleed Kamal | Tehamee Raheel | Abdul Ahad
Object Detection | Android | YOLO | Python

 
