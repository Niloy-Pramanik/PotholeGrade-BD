"""
YOLOv8 Segmentation Model Training Module for PotholeGrade-BD.

Fine-tunes a pre-trained YOLOv8 nano segmentation model on a custom
pothole dataset using transfer learning.
"""

from pathlib import Path
from ultralytics import YOLO


def train_yolo_model(
    data_yaml: str = "data/dataset_yolo/data.yaml",
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: int = 0
) -> None:
    """
    Fine-tunes a pre-trained YOLOv8 segmentation model on custom data.

    Loads a pre-trained YOLOv8 nano segmentation model and fine-tunes it
    on the custom pothole dataset defined in data.yaml. Trained weights
    are saved to runs/segment/train/weights/best.pt.

    Args:
        data_yaml (str): Path to Roboflow-exported data.yaml file.
            Default is 'data/dataset_yolo/data.yaml'.
        model_name (str): Name of pre-trained model to load.
            Default is 'yolov8n-seg.pt' (nano, fastest).
        epochs (int): Number of training epochs. Default is 50.
        imgsz (int): Input image size. Default is 640.
        batch_size (int): Batch size for training. Default is 16.
        device (int): GPU device ID. Use 0 for first GPU, or -1 for CPU.
            Default is 0.

    Returns:
        None

    Raises:
        FileNotFoundError: If data.yaml file does not exist.

    Example:
        >>> train_yolo_model(
        ...     data_yaml="data/dataset_yolo/data.yaml",
        ...     epochs=50,
        ...     batch_size=16
        ... )

    Notes:
        - Requires ultralytics package: pip install ultralytics
        - First run will download yolov8n-seg.pt (~15 MB)
        - Training logs are saved in runs/segment/train/
    """
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ data.yaml not found at '{data_yaml}'.")

    print("🤖 Loading pre-trained YOLOv8 model: {}".format(model_name))
    model = YOLO(model_name)

    print(f"🚀 Starting transfer learning training with dataset: {data_yaml}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Image size: {imgsz}x{imgsz}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Device: {'GPU' if device >= 0 else 'CPU'}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=10,
        save=True,
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"📊 Best weights saved to: runs/segment/train/weights/best.pt")


if __name__ == "__main__":
    train_yolo_model(
        data_yaml="data/dataset_yolo/data.yaml",
        epochs=50,
        batch_size=16
    )
