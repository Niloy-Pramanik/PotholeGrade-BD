"""
Main Inference Pipeline for PotholeGrade-BD.

Combines YOLOv8 detection, DIP analysis, and RPS scoring to process
video frames and visualize pothole detection results in real-time.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO

from dip_engine import PotholeDIPEngine
from rps_logic import calculate_rps


class PotholeInferencePipeline:
    """
    End-to-end inference pipeline for pothole detection and analysis.

    Combines pre-trained YOLOv8 segmentation model with digital image
    processing and priority scoring to detect, analyze, and visualize
    potholes in video frames.

    Attributes:
        model (YOLO): Loaded YOLOv8 segmentation model.
        dip_engine (PotholeDIPEngine): DIP engine for metric calculation.
        confidence_threshold (float): Minimum confidence for detections.

    Example:
        >>> pipeline = PotholeInferencePipeline(
        ...     model_path="runs/segment/train/weights/best.pt"
        ... )
        >>> pipeline.process_video("test_video.mp4")
    """

    def __init__(
        self,
        model_path: str = "runs/segment/train/weights/best.pt",
        confidence_threshold: float = 0.5
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            model_path (str): Path to trained YOLOv8 model weights.
                Default is 'runs/segment/train/weights/best.pt'.
            confidence_threshold (float): Minimum confidence score for detections.
                Default is 0.5.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"❌ Model weights not found at '{model_path}'")

        print(f"🤖 Loading YOLO model from: {model_path}")
        self.model: YOLO = YOLO(model_path)
        self.dip_engine: PotholeDIPEngine = PotholeDIPEngine()
        self.confidence_threshold: float = confidence_threshold

        print(f"✅ Model loaded successfully")
        print(f"   Confidence threshold: {confidence_threshold}")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True
    ) -> None:
        """
        Process a video file and detect/analyze potholes in real-time.

        Reads frames from the input video, runs YOLOv8 inference, calculates
        pothole metrics via DIP, computes RPS scores, and displays results
        with annotations.

        Args:
            video_path (str): Path to input video file (.mp4 or similar).
            output_path (str, optional): Path to save output video with annotations.
                If None, output is not saved. Default is None.
            display (bool): Whether to display frames in real-time.
                Default is True.

        Returns:
            None

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video cannot be opened.

        Example:
            >>> pipeline.process_video(
            ...     video_path="test.mp4",
            ...     output_path="output.mp4",
            ...     display=True
            ... )
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"❌ Cannot open video file: {video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS)
        frame_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count: int = 0
        pothole_detections: int = 0

        print(f"\n▶️  Processing video: {video_path}")
        print(f"   Resolution: {frame_width}x{frame_height} @ {fps:.1f} fps")
        print(f"   Total frames: {total_frames}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Run YOLO inference and get annotated frame
            annotated_frame, detections_in_frame = self._process_frame(frame)
            pothole_detections += detections_in_frame

            # Display frame
            if display:
                cv2.imshow("PotholeGrade-BD Inference", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏸️  Inference stopped by user (q pressed)")
                    break

            # Write to output video
            if writer:
                writer.write(annotated_frame)

            if frame_count % 30 == 0:
                print(f"   Processed {frame_count}/{total_frames} frames... ({pothole_detections} potholes detected)")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Processing complete!")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Total potholes detected: {pothole_detections}")
        if output_path:
            print(f"   📁 Output saved: {output_path}")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process a single frame: detect potholes, calculate metrics, annotate.

        Args:
            frame (np.ndarray): RGB/BGR image frame.

        Returns:
            Tuple[np.ndarray, int]: Annotated frame and count of detections.
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)

        annotated_frame: np.ndarray = frame.copy()
        detection_count: int = 0

        # Process each detection
        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.xy
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names

            for mask, conf, class_id in zip(masks, confidences, class_ids):
                # Filter by confidence
                if conf < self.confidence_threshold:
                    continue

                detection_count += 1

                # Get class name
                class_name: str = class_names.get(int(class_id), "Unknown")

                # Prepare polygon
                polygon = mask.astype(np.int32)

                try:
                    # Calculate metrics via DIP engine
                    metrics = self.dip_engine.calculate_metrics(frame, polygon)

                    # Calculate RPS
                    rps: int = calculate_rps(metrics.volume_kg, class_name)

                    # Determine color based on RPS (Green -> Yellow -> Red)
                    if rps <= 2:
                        color: Tuple[int, int, int] = (0, 255, 0)  # Green
                    elif rps == 3:
                        color = (0, 165, 255)  # Orange
                    else:  # RPS 4 or 5
                        color = (0, 0, 255)  # Red

                    # Draw polygon
                    cv2.drawContours(annotated_frame, [polygon], 0, color, 2)

                    # Create text label
                    label: str = (
                        f"{class_name} | "
                        f"Vol: {metrics.volume_kg:.1f} kg | "
                        f"RPS: {rps}"
                    )

                    # Get bounding box for text placement
                    x_min, y_min = polygon.min(axis=0)
                    text_pos: Tuple[int, int] = (x_min, max(10, y_min - 10))

                    # Draw text background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (text_pos[0] - 5, text_pos[1] - text_height - 5),
                        (text_pos[0] + text_width + 5, text_pos[1] + baseline + 5),
                        (0, 0, 0),
                        -1
                    )

                    # Draw text label
                    cv2.putText(
                        annotated_frame,
                        label,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                except Exception as e:
                    print(f"⚠️  Error processing detection: {e}")
                    continue

        return annotated_frame, detection_count


def main() -> None:
    """
    Main entry point for the inference pipeline.

    Expects a test video file at 'data/test_video.mp4'.
    """
    # Initialize pipeline
    try:
        pipeline = PotholeInferencePipeline(
            model_path="runs/segment/train/weights/best.pt",
            confidence_threshold=0.5
        )
    except FileNotFoundError:
        print("❌ Error: Model weights not found.")
        print("   Please train the model first using: python src/02_train_yolo.py")
        return

    # Process video
    test_video: str = "data/test_video.mp4"

    if not Path(test_video).exists():
        print(f"❌ Error: Test video not found at '{test_video}'")
        print("   Please provide a test video file in data/test_video.mp4")
        return

    pipeline.process_video(
        video_path=test_video,
        output_path="output_annotated.mp4",
        display=True
    )


if __name__ == "__main__":
    main()
