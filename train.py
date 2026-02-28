from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train on your dataset
model.train(
    data="data/data.yaml",
    epochs=75,
    imgsz=640,
    batch=8
)
