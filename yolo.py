import torch
import cv2

# Load pre-trained YOLOv5 small model (fast + accurate)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define obstacle categories you care about (can be expanded)
obstacle_classes = ['person', 'car', 'truck', 'bus', 'motorbike', 'bicycle']

# Open camera (0 = default webcam, or replace with video file path)
cap = cv2.VideoCapture("C:/Users/chsai/Downloads/archive (6)/4K Road traffic video for object detection and tracking - free download now.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Parse results
    detections = results.pandas().xyxy[0]  # bounding boxes, labels, confidence
    obstacle_detected = False

    for _, row in detections.iterrows():
        label = row['name']
        conf = row['confidence']
        if label in obstacle_classes and conf > 0.5:
            obstacle_detected = True
            # Draw bounding box
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show obstacle flag
    if obstacle_detected:
        cv2.putText(frame, "Obstacle Detected!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print("Obstacle Detected")
    else:
        cv2.putText(frame, "Clear Path", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("Obstacle Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
