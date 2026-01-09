import socket
import cv2
import numpy as np
from ultralytics import YOLO
import struct
import json

# Load YOLO model
model = YOLO("yolov8n.pt")  # pretrained COCO model or your custom model

# Server setup
HOST = "0.0.0.0"
PORT = 9999
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"Server listening on {HOST}:{PORT}...")
conn, addr = server.accept()
print(f"Connected by {addr}")

# --- Create a normal, resizable window ---
WINDOW_NAME = "YOLO Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

def recv_all(sock, length):
    data = b""
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            return None
        data += more
    return data

try:
    while True:
        # Receive frame size from phone
        size_bytes = recv_all(conn, 4)
        if not size_bytes:
            print("Client disconnected")
            break
        frame_size = struct.unpack('>I', size_bytes)[0]

        # Receive frame data from phone
        frame_data = recv_all(conn, frame_size)
        if not frame_data:
            print("Frame data not received")
            break

        # Decode JPEG
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # --- The image from Android is now correctly oriented, no rotation needed ---

        # YOLO detection
        results = model(img)
        
        # --- Prepare detection data to send back to phone ---
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = model.names[cls]
                
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

        # --- Send detection data back to the phone ---
        detections_json = json.dumps(detections)
        json_bytes = detections_json.encode('utf-8')
        json_size_bytes = struct.pack('>I', len(json_bytes))
        
        conn.sendall(json_size_bytes)
        conn.sendall(json_bytes)

        # Annotated frame for display on laptop
        annotated_frame = results[0].plot()

        cv2.imshow(WINDOW_NAME, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("Error:", e)
finally:
    conn.close()
    server.close()
    cv2.destroyAllWindows()
