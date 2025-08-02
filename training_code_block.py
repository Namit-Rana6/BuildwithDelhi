from ultralytics import YOLO

model = YOLO("yolov8x.pt")  # Best performing pretrained YOLOv8 model

model.train(
    data="/tmp/hackathon_dataset/HackByte_Dataset/yolo_params.yaml",   # Custom YAML file with paths + class names
    epochs=100,               # More epochs to ensure full learning
    imgsz=896,                # High-res improves detection accuracy
    batch=8,                  # Safe for T4 GPU, increase if more VRAM
    optimizer="AdamW",        # Best general-purpose optimizer
    lr0=1e-4,                 # Low learning rate = better generalization
    weight_decay=0.001,       # Regularization to prevent overfitting
    patience=15,              # Early stopping if no val improvement
    dropout=0.10,             # Prevents overfitting (important for small datasets)
    label_smoothing=0.05,     # Helps avoid overconfidence on noisy labels

    # ⚙️ Augmentations — helps avoid overfitting
    mosaic=0.4,               # Reduced to avoid unrealistic augmentations
    mixup=0.15,               # Light mixup for generalization
    hsv_h=0.015,              # Slight color hue change
    hsv_s=0.6,                # Slight saturation shift
    hsv_v=0.4,                # Slight brightness variation
    translate=0.1,
    scale=0.5,

    project="hackbyte-final",
    name="yolov8x-best",
    save=True,
    val=True                 # Ensure validation during training
)
