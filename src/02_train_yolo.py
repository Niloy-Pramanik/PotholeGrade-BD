"""
YOLOv8 Object Detection Model Training Module for PotholeGrade-BD PoC.

Fine-tunes a pre-trained YOLOv8 nano object detection model on the RDD2020
bounding box dataset for proof of concept. This validates our training pipeline
before collecting custom segmentation data for production.
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
    Fine-tunes a pre-trained YOLOv8 object detection model on RDD2020 dataset.

    This is a Proof of Concept (PoC) implementation that trains on the RDD2020
    bounding box dataset to validate the training pipeline. Once validated, the
    pipeline will be used with custom segmentation data collected from Bangladesh
    roads. Trained weights are saved to runs/detect/rdd_poc_model/weights/best.pt.

    Args:
        data_yaml (str): Path to RDD2020 data.yaml file.
            Default is 'data/rdd_sample/data.yaml'.
        model_name (str): Name of pre-trained object detection model to load.
            Default is 'yolov8n.pt' (nano, fastest for PoC).
            Note: Using bounding box model, not segmentation for this PoC.
        epochs (int): Number of training epochs. Default is 20 (fast PoC training).
        imgsz (int): Input image size. Default is 640 (standard for YOLO).
        batch_size (int): Batch size for training. Default is 16.
        device (int): GPU device ID. Use 0 for first GPU, or -1 for CPU.
            Default is 0.
        project (str): Project directory for saving results.
            Default is 'runs/detect'.
        name (str): Name of the training run for result subdirectory.
            Default is 'rdd_poc_model'.

    Returns:
        None

    Raises:
        FileNotFoundError: If data.yaml file does not exist.

    Example:
        >>> train_yolo_model(
        ...     data_yaml="data/rdd_sample/data.yaml",
        ...     epochs=20,
        ...     batch_size=16
        ... )

    Notes:
        - Requires ultralytics package: pip install ultralytics
        - First run will download yolov8n.pt (~6.3 MB)
        - This is a PoC using RDD2020 bounding box data
        - Production version will use custom segmentation data
        - Training logs are saved in runs/detect/rdd_poc_model/
        - Best weights saved to: runs/detect/rdd_poc_model/weights/best.pt
    """
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ data.yaml not found at '{data_yaml}'.")

    # Smart device detection for macOS M1/M2/M3 and other platforms
    import torch
    if isinstance(device, int) and device >= 0:
        # Check for M1/M2/M3 GPU (Metal Performance Shaders)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
            device_name = "Apple M-Series GPU (MPS)"
        # Check for NVIDIA CUDA
        elif torch.cuda.is_available():
            device = 0
            device_name = "NVIDIA CUDA GPU"
        # Fall back to CPU
        else:
            device = "cpu"
            device_name = "CPU"
    else:
        device_name = "CPU"

    print("=" * 70)
    print("🏗️  PotholeGrade-BD: Proof of Concept (PoC) Training")
    print("=" * 70)
    print(f"\n📚 Dataset: RDD2020 (Bounding Box Detection)")
    print(f"🎯 Purpose: Validate training pipeline before production deployment")
    print(f"🔄 Next Phase: Transition to custom segmentation data from Bangladesh roads\n")

    print("🤖 Loading pre-trained YOLOv8 object detection model: {}".format(model_name))
    model = YOLO(model_name)

    print(f"\n🚀 Starting transfer learning training with RDD2020 dataset")
    print(f"   • Dataset path: {data_yaml}")
    print(f"   • Epochs: {epochs} (PoC - fast training)")
    print(f"   • Image size: {imgsz}x{imgsz}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Device: {device_name} ({device})")
    print(f"   • Project directory: {project}")
    print(f"   • Run name: {name}\n")

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

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"📊 Best weights saved to: {project}/{name}/weights/best.pt")
    print(f"📈 Training plots saved to: {project}/{name}/")
    print(f"\n💡 PoC Validation Complete!")
    print(f"🎓 Next Step: Collect custom pothole segmentation data from Bangladesh roads")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_yolo_model(
        data_yaml="data/rdd_sample/data.yaml",
        model_name="yolov8n.pt",
        epochs=20,
        batch_size=16,
        project="runs/detect",
        name="rdd_poc_model"
    )
