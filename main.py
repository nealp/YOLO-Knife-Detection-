from ultralytics import YOLO
import cv2

# Load the YOLOv10 model
model = YOLO('my_model.pt')  # path to your custom-trained model

# Open webcam or video
cap = cv2.VideoCapture('videos/test1.mp4')  # Replace with 'video.mp4' for a video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()  # includes bounding boxes, labels, etc.

    # Display the annotated frame
    cv2.imshow("YOLOv10 Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
