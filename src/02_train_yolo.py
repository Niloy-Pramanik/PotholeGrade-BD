"""
YOLOv8 Object Detection Model Training Module for PotholeGrade-BD PoC.

Fine-tunes a pre-trained YOLOv8 nano object detection model on the RDD2020
bounding box dataset. Validates training pipeline before custom data collection.
"""

from pathlib import Path
from ultralytics import YOLO


def train_yolo_model(
    data_yaml: str = "data/rdd_sample/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 20,
    imgsz: int = 640,
    batch_size: int = 16,
    device: int | str = 0,
    project: str = "runs/detect",
    name: str = "rdd_poc_model"
) -> None:
    """
    Fine-tune YOLOv8 object detection model on RDD2020 dataset.

    Args:
        data_yaml: Path to data.yaml file (default: data/rdd_sample/data.yaml)
        model_name: Pre-trained model name (default: yolov8n.pt)
        epochs: Number of training epochs (default: 20)
        imgsz: Input image size (default: 640)
        batch_size: Batch size for training (default: 16)
        device: GPU device ID or 'cpu' (default: 0)
        project: Project directory for results (default: runs/detect)
        name: Training run name (default: rdd_poc_model)

    Raises:
        FileNotFoundError: If data.yaml file not found

    Returns:
        None
    """
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found at '{data_yaml}'")

    # Auto-detect best device
    import torch
    if isinstance(device, int) and device >= 0:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
            device_name = "Apple M-Series GPU"
        elif torch.cuda.is_available():
            device = 0
            device_name = "NVIDIA GPU"
        else:
            device = "cpu"
            device_name = "CPU"
    else:
        device_name = "CPU"

    print("PotholeGrade-BD: YOLOv8 Training (RDD2020 PoC)")
    print("-" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Dataset: {data_yaml}")
    print("-" * 60 + "\n")

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=5,
        save=True,
        verbose=True
    )

    print("\n" + "-" * 60)
    print("Training complete")
    print(f"Best weights: {project}/{name}/weights/best.pt")
    print(f"Results: {project}/{name}/")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    train_yolo_model(
        data_yaml="data/rdd_sample/data.yaml",
        model_name="yolov8n.pt",
        epochs=20,
        batch_size=16,
        project="runs/detect",
        name="rdd_poc_model"
    )
