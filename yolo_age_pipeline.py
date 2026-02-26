import cv2
from ultralytics import YOLO
from age_inference import predict_age

# Load YOLO (person detector)
yolo = YOLO("yolov8n.pt")   # pretrained COCO

# Open image or video
cap = cv2.VideoCapture("input.mp4")  # or 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, conf=0.4, classes=[0])  # class 0 = person

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            age_label = predict_age(person_crop)

            # Draw bbox + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                age_label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

    cv2.imshow("YOLO + Age Group", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
