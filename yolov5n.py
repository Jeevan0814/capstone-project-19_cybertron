import cv2
import torch
import json
import time

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Open video capture (0 = default camera, or replace with video file)
cap = cv2.VideoCapture("C:/Users/chsai/Downloads/archive (6)/4K Road traffic video for object detection and tracking - free download now.mp4")

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = model(frame)

    detections = []
    filtered_detections = []

    for *bbox, conf, cls in results.xyxy[0].tolist():
        label = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Determine position in frame
        if cx < frame.shape[1] // 3:
            position = "left"
        elif cx > 2 * frame.shape[1] // 3:
            position = "right"
        else:
            position = "center"

        detection = {
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2],
            "position": position,
            "size": {"w": w, "h": h}
        }

        detections.append(detection)

        # Distance filter (only log "near" obstacles)
        if h > 80 or w > 80:  # adjust threshold if needed
            detection["near"] = True
            filtered_detections.append(detection)
        else:
            detection["near"] = False

        # Draw bounding box (all detections, not just near)
        color = (0, 255, 0) if detection["near"] else (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save only near detections to JSON
    log_data = {
        "ts": time.time(),
        "frame": frame_id,
        "detections": filtered_detections,
        "obstacle": len(filtered_detections) > 0
    }

    with open("detections.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # Show video feed
    cv2.imshow("Obstacle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
