"""
Inference Pipeline - Detect potholes and calculate metrics.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO

from dip_engine import PotholeDIPEngine
from rps_logic import calculate_rps


class PotholeInferencePipeline:
    """End-to-end inference: YOLOv8 + DIP + RPS scoring."""

    def __init__(
        self,
        model_path: str = "runs/segment/train/weights/best.pt",
        confidence_threshold: float = 0.5
    ) -> None:
        """
        Initialize pipeline.

        Args:
            model_path: Path to trained YOLOv8 weights
            confidence_threshold: Minimum confidence for detections

        Raises:
            FileNotFoundError: If model file not found
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading YOLO model from: {model_path}")
        self.model: YOLO = YOLO(model_path)
        self.dip_engine: PotholeDIPEngine = PotholeDIPEngine()
        self.confidence_threshold: float = confidence_threshold
        print(f"Model loaded. Confidence threshold: {confidence_threshold}")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True
    ) -> None:
        """
        Process video and detect potholes.

        Args:
            video_path: Input video file path
            output_path: Output video path (optional)
            display: Show frames in real-time

        Raises:
            FileNotFoundError: If video not found
            ValueError: If video cannot be opened
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

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

        print(f"\nProcessing: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height} @ {fps:.1f} fps")
        print(f"Total frames: {total_frames}\n")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            annotated_frame, detections = self._process_frame(frame)
            pothole_detections += detections

            if display:
                cv2.imshow("PotholeGrade-BD", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopped by user")
                    break

            if writer:
                writer.write(annotated_frame)

            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} ({pothole_detections} potholes)")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\nComplete. Frames: {frame_count}, Potholes: {pothole_detections}")
        if output_path:
            print(f"Output: {output_path}")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process single frame and return annotated output."""
        results = self.model(frame, verbose=False)
        annotated_frame: np.ndarray = frame.copy()
        detection_count: int = 0

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.xy
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names

            for mask, conf, class_id in zip(masks, confidences, class_ids):
                if conf < self.confidence_threshold:
                    continue

                detection_count += 1
                class_name: str = class_names.get(int(class_id), "Unknown")
                polygon = mask.astype(np.int32)

                try:
                    metrics = self.dip_engine.calculate_metrics(frame, polygon)
                    rps: int = calculate_rps(metrics.volume_kg, class_name)

                    # Color by RPS: Green (1-2), Orange (3), Red (4-5)
                    if rps <= 2:
                        color: Tuple[int, int, int] = (0, 255, 0)
                    elif rps == 3:
                        color = (0, 165, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.drawContours(annotated_frame, [polygon], 0, color, 2)

                    label: str = f"{class_name} | Vol: {metrics.volume_kg:.1f} kg | RPS: {rps}"
                    x_min, y_min = polygon.min(axis=0)
                    text_pos: Tuple[int, int] = (x_min, max(10, y_min - 10))

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
                    print(f"Error: {e}")
                    continue

        return annotated_frame, detection_count


def main() -> None:
    """Main entry point."""
    try:
        pipeline = PotholeInferencePipeline(
            model_path="runs/segment/train/weights/best.pt",
            confidence_threshold=0.5
        )
    except FileNotFoundError:
        print("Error: Model not found. Run: python src/02_train_yolo.py")
        return

    test_video: str = "data/test_video.mp4"

    if not Path(test_video).exists():
        print(f"Error: Video not found at {test_video}")
        return

    pipeline.process_video(
        video_path=test_video,
        output_path="output_annotated.mp4",
        display=True
    )


if __name__ == "__main__":
    main()