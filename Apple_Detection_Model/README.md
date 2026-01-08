# 🍎 Apple Quality Detection – YOLOv8m

A computer vision system built using Ultralytics, train images using Roboflow on YOLOv8m to classify apples into four quality categories:
- Red Apple
- Green Apple
- Used Apple
- Rotten Apple

### 🚀 Project Overview

This project aims to automate fruit quality inspection using a lightweight custom model trained on YOLOv8 model. The system can run efficiently even on low-resource devices while maintaining strong detection accuracy.

The model is trained on a custom dataset and deployed with a Streamlit web interface for easy testing.

### 🧠 Features
- Custom object detection model using custom model trained on YOLOv8m.
- Detects 4 apple categories in images or video
- Real-time inference using webcam
- Clean and interactive Streamlit UI
- Compatible with CPU and small GPUs (2GB+)

### 📸 Demo

<img width="277" height="182" alt="Detected Image 1" src="https://github.com/user-attachments/assets/75b932a8-e2ad-40e5-983f-7eed176e3281" />
<img width="277" height="122" alt="Detected Image 2" src="https://github.com/user-attachments/assets/7086d147-b9a7-41c4-ae76-91f988aad10d" />
<img width="279" height="276" alt="Detected Image 3" src="https://github.com/user-attachments/assets/54eda85d-2acb-40b9-8752-258086afa2d3" />


🛠️ Technologies Used
- Python
- Roboflow
- Ultralytics YOLOv8
- OpenCV
- Streamlit

### ▶️ Run the Project
1. Install packages
```bash
pip install roboflow ultralytics streamlit opencv-python
```

3. Run Streamlit app
```bash
streamlit run app.py
```

### 🤖 Training the Model
```bash
yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640
```

### ⭐ Contribute

Pull requests and suggestions are welcome!

### 📬 Contact

For queries or collaboration:
Muhammad Waleed Kamal – [LinkedIn Profile](https://www.linkedin.com/in/muhammad-waleed-kamal-3910422b3/)
