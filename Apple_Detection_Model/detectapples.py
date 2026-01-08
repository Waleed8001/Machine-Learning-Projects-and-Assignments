from ultralytics import YOLO
import cv2
import streamlit as st
import tempfile
import os

# Set Page Configuration
st.set_page_config(
    page_title="Detect Apples",
    layout="wide"
)

st.header("🍎 Apple Quality Detection – Real-Time Classifier")

# Intro of out Work.
cola, colb = st.columns(2)

with cola:
    st.write('''🍎 Apple Quality Detection is an interactive Streamlit web app that uses a custom YOLOv8n model to intelligently detect and classify apples into four categories — 

🍏 green apples

🍎 red apples

🍂 used/damaged apples

🟤 rotten apples. 

The app offers a smooth experience where users can effortlessly ✨ upload images or 🎥 use their webcam for real-time detection, while automatically displaying 🔲 bounding boxes and 📊 confidence scores on the results. Built using powerful technologies like Ultralytics YOLOv8n, Streamlit, OpenCV, and Python, the system is optimized to run ⚡ efficiently even on low-spec devices. With its clean UI and fast inference, this web app makes it easy to explore, test, and visualize apple quality directly from your browser. 🍏🚀''')
    
with colb:
    st.image("buspic.jpg")
    
# For Upload Images from Streamlit UI.
apple_image = st.file_uploader(
    "Enter a Picture", 
    type=["jpeg", "jpg", "png", "webp"], 
    accept_multiple_files=False,
)

# Merge Temporary path and image name.
if apple_image:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, apple_image.name)
    with open(path, "wb") as f:
        f.write(apple_image.getvalue())


# Import Trained Model.
model = YOLO("DetectApples.pt")

# file_path = "apples10.jpeg"
# results = model.predict(file_path, conf=0.3, imgsz=640)

if apple_image:
    # Create Columns in streamlit.
    col1, col2, col3 = st.columns(3, vertical_alignment="center")

    with col1:
        st.image(apple_image, caption="Real Image", use_container_width=True)

    with col2:
        st.write("===========================================>")
        st.write("===========================================>")
        st.write("===========================================>")
        st.write("===========================================>")

    with col3:
        results = model.predict(path, conf=0.5, imgsz=640)

        plotted = results[0].plot()[:,:,::-1]
        st.image(plotted, caption="Detection Image", use_container_width=True)

    # Store names of all Detected Images.
    boxes = results[0].boxes
    names = []

    for box in boxes:
        names.append(results[0].names[int(box.cls)])

    print(names)

    # Count all Detected Images.
    count_apples = {}
    count = 1

    for i in names:
        if i not in count_apples:
            count_apples[i] = count

        else:
            count_apples[i] += 1

    # Show all Detected Images.
    for key, value in count_apples.items():
        st.write(f"{key}: {value}")

    



# For working with Live Detection from Webcam
# results = model(source=0, conf=0.20, show=True) # For Live detection from Computer/Laptop default Camera
# results = model(source=1, conf=0.20, show=True) # For Live detection from Mobile using DroidCam

# print(results)

# # --------------------------- For working with CV2 -----------------------------------
# # The data of result is in the tensor format which we can get from boxes
# for result in results:
#     boxes = result.boxes

# # print(boxes)

# image = cv2.imread(file_path)

# for result in results:
#     xywh = result.boxes.xywh  # center-x, center-y, width, height
#     xywhn = result.boxes.xywhn  # normalized
#     xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
#     xyxyn = result.boxes.xyxyn  # normalized
#     names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
#     confs = result.boxes.conf  # confidence score of each box

# # Now, you iterate like this to draw your bounding boxes and put text
# for myxyxy, conf, cls in zip(xyxy, confs, names):
#     label = cls
#     xyxy = [int(x) for x in myxyxy]  # Convert tensor to int
#     if label == "Green Juicy Apple":
#         cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
#         cv2.putText(image, f'{label}:', (xyxy[0], xyxy[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
#         cv2.putText(image, f'{conf:.2f}', (xyxy[0], xyxy[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
#     elif label == "Rotten Apple":
#         cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
#         cv2.putText(image, f'{label}:', (xyxy[0], xyxy[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#         cv2.putText(image, f'{conf:.2f}', (xyxy[0], xyxy[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
#     elif label == "Red Apple":
#         cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
#         cv2.putText(image, f'{label}:', (xyxy[0], xyxy[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
#         cv2.putText(image, f'{conf:.2f}', (xyxy[0], xyxy[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
#     elif label == "Used Apple":
#         cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 0), 2)
#         cv2.putText(image, f'{label}:', (xyxy[0], xyxy[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
#         cv2.putText(image, f'{conf:.2f}', (xyxy[0], xyxy[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# cv2.imshow('Object Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(xywh)
# print(xywhn)
# print(xyxy)
# print(xyxyn)
# print(names)
# print(confs)

# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     # result.show()  # display to screen
#     # result.save(filename="result.jpg")  # save to disk

# print(boxes)
# print(masks)
# print(keypoints)
# print(probs)
# print(obb)
# print(confs)
