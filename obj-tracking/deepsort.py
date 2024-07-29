from deepsort import DeepSortTracker

class DeepSORTTracker:
    def __init__(self, max_age=70, n_init=3, nn_budget=100, model_path='deep_sort_cpp/model_data/mars-small128.pb'):
        self.tracker = DeepSORTTracker(max_age, n_init, nn_budget, model_path)

    def update(self, detections):
        boxes = np.array([d[0] for d in detections])
        confidences = np.array([d[1] for d in detections])
        class_ids = np.array([d[2] for d in detections])

        features = None  # Replace with actual feature extraction if needed

        tracked_objects = self.tracker.update(boxes, confidences, class_ids, features)
        return tracked_objects
    