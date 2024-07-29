from ultralytics import YOLO
import yaml

model=YOLO("FastSAM-s.pt")

results=model.train(data=r"/home/adithyadk/Desktop/model-ai/dataset/dataset4/010101.v1i.yolov9/data.yaml", epochs=100, patience=100)