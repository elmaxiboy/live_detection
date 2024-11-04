import cv2
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on the source (0 for default webcam)
results = model(source=0, stream=True)  # generator of Results objects

# Open a display window
window_name = "YOLO Real-Time Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    for result in results:
        # Get the frame with detections
        frame = result.orig_img  # The original image/frame

        # Draw bounding boxes and labels
        for box in result.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            label = box.cls  # The label/class of the detected object
            confidence = box.conf  # Confidence score
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display label and confidence score
            text = f"{label}: {float(confidence):.2f}"

            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with detections
        cv2.imshow(window_name, frame)

        # Press 'q' to quit the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Release resources
    cv2.destroyAllWindows()
