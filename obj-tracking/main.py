import cv2
from yolo import YOLO
from deepsort import DeepSortTracker
def main():
    # Initialize YOLO model
    yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')

    # Initialize DeepSORT tracker
    tracker = DeepSortTracker()

    # Load video or camera
    cap = cv2.VideoCapture(0)  # Use camera 0 for live video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects using YOLO
        boxes, confidence ,classids= yolo.detect_objects(frame)
        detections = list(zip(boxes, confidences, class_ids))

        # Update tracker with detections
        tracked_objects = tracker.update(detections)

        # Draw bounding boxes and labels for tracked objects
        for track in tracked_objects:
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow('DeepSORT with YOLO', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()