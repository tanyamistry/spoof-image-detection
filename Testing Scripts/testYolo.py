from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Use webcam (try index 0 first; use 1 if you have external camera)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Check if webcam opened successfully
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# Load YOLOv8 model (nano, small, or any version you downloaded)
model = YOLO("../models/yolov8n.pt")  # Adjust path if needed

# COCO classes
classNames = model.names

# Loop through webcam frames
while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from webcam")
        break

    # Run YOLOv8 inference
    results = model(img, stream=True, verbose=False)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coords
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Draw on frame
            cvzone.putTextRect(img, f'{class_name} {conf}', (x1, y1 - 10), scale=1, thickness=1)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9)

    cv2.imshow("YOLOv8 Webcam Detection", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
