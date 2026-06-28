from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data=r"C:\Users\Johaan Liju\Downloads\label_dataset\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",      # GPU (use "cpu" if no GPU)
    workers=8
)

# Save automatically
print("Training complete")