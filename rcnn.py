import os
import yaml
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
import cv2

# Define the CustomDataset class
class CustomDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transform=None):
        super(CustomDataset, self).__init__(img_folder, ann_file, transform=transform)

    def __getitem__(self, idx):
        img, target = super(CustomDataset, self).__getitem__(idx)
        target = {
            'boxes': torch.tensor([ann['bbox'] for ann in target], dtype=torch.float32),
            'labels': torch.tensor([ann['category_id'] for ann in target], dtype=torch.int64)
        }
        return img, target

# Define the model
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.body.layer4.parameters():
        param.requires_grad = True
    return model

# Load YAML configuration from a specific path
def load_yaml_config(filepath='/home/adithyadk/Desktop/model-ai/dataset/datset2/data.yaml'):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

# Get transformations
def get_transform():
    return T.Compose([
        T.ToTensor(),
    ])

# Main function
def main():
    # Load configuration
    config = load_yaml_config('/home/adithyadk/Desktop/model-ai/dataset/datset2/data.yaml')
    train_img_folder = config['train']['images']
    train_ann_file = config['train']['annotations']
    val_img_folder = config['val']['images']
    val_ann_file = config['val']['annotations']
    test_img_folder = config['test']['images']
    test_ann_file = config['test']['annotations']
    num_classes = config['num_classes']


    # Create dataset instances
    train_dataset = CustomDataset(train_img_folder, train_ann_file, transform=get_transform())
    val_dataset = CustomDataset(val_img_folder, val_ann_file, transform=get_transform())
    test_dataset = CustomDataset(test_img_folder, test_ann_file, transform=get_transform())

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    # Instantiate model and optimizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training function
    def train_one_epoch(model, data_loader, optimizer, device):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    # Evaluation function
    def evaluate(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                prediction = model(images)
                # Evaluate predictions here (e.g., compute IoU, accuracy, etc.)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        evaluate(model, val_loader, device)

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    # Load the model for inference
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Inference function
    def test(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                predictions = model(images)
                # Process and save predictions for test images

    # Perform inference on test data
    test(model, test_loader, device)

    # Live video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(frame_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(img_tensor)
        boxes = prediction[0]['boxes'].cpu().numpy()
        masks = prediction[0]['masks'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                mask = masks[i][0]
                frame[mask > 0.5] = [0, 255, 0]
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
