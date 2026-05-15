"""
Frame Extraction Module - Extract frames from dashcam videos at 1 fps.
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_videos(
    video_dir: str = "data/raw_videos",
    output_dir: str = "data/images",
    fps_target: int = 1
) -> None:
    """
    Extract frames from MP4 videos at target frame rate.

    Args:
        video_dir: Directory containing .mp4 videos (default: data/raw_videos)
        output_dir: Output directory for extracted frames (default: data/images)
        fps_target: Frames per second to extract (default: 1)

    Raises:
        FileNotFoundError: If video_dir does not exist
        ValueError: If fps_target <= 0
    """
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory '{video_dir}' not found")

    if fps_target <= 0:
        raise ValueError(f"fps_target must be > 0, got {fps_target}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    video_files = list(Path(video_dir).glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in '{video_dir}'")
        return

    frame_counter: int = 0

    for video_file in video_files:
        print(f"\nProcessing: {video_file.name}")

        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"Error: Could not open '{video_file.name}'. Skipping...")
            continue

        fps_video: float = cap.get(cv2.CAP_PROP_FPS)
        frame_interval: int = max(1, int(fps_video / fps_target))
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx: int = 0

        with tqdm(
            total=total_frames,
            desc=f"Extracting @ {fps_target} fps",
            unit="frame"
        ) as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    output_path = os.path.join(
                        output_dir,
                        f"frame_{frame_counter:06d}.jpg"
                    )
                    cv2.imwrite(output_path, frame)
                    frame_counter += 1

                frame_idx += 1
                pbar.update(1)

        cap.release()

    print(f"\nExtraction complete. Total frames: {frame_counter}")


if __name__ == "__main__":
    extract_frames_from_videos(
        video_dir="data/raw_videos",
        output_dir="data/images",
        fps_target=1
    )
