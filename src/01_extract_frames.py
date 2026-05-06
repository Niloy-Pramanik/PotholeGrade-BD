"""
Frame Extraction Module for PotholeGrade-BD.

Extracts frames from raw smartphone dashcam videos at 1 frame per second
and saves them as JPEG images for annotation and training.
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
    Extracts frames from all MP4 videos in a directory at a specified frame rate.

    This function iterates through all .mp4 files in video_dir, and extracts
    frames at the target frame rate (default 1 fps). Frames are saved as JPEG
    images with sequential numbering in the output directory.

    Args:
        video_dir (str): Path to directory containing .mp4 videos.
            Default is 'data/raw_videos'.
        output_dir (str): Path to directory where extracted frames will be saved.
            Default is 'data/images'. Directory is created if it doesn't exist.
        fps_target (int): Target frames per second to extract. Default is 1.

    Returns:
        None

    Raises:
        FileNotFoundError: If video_dir does not exist.
        ValueError: If fps_target is <= 0.

    Example:
        >>> extract_frames_from_videos(
        ...     video_dir="data/raw_videos",
        ...     output_dir="data/images",
        ...     fps_target=1
        ... )
    """
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory '{video_dir}' not found.")

    if fps_target <= 0:
        raise ValueError(f"fps_target must be > 0, got {fps_target}.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    video_files = list(Path(video_dir).glob("*.mp4"))

    if not video_files:
        print(f"⚠️  No .mp4 files found in '{video_dir}'.")
        return

    frame_counter: int = 0

    for video_file in video_files:
        print(f"\n📹 Processing video: {video_file.name}")

        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"❌ Error: Could not open video '{video_file.name}'. Skipping...")
            continue

        fps_video: float = cap.get(cv2.CAP_PROP_FPS)
        frame_interval: int = max(1, int(fps_video / fps_target))
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx: int = 0

        with tqdm(
            total=total_frames,
            desc=f"Extracting @ {fps_target} fps",
            unit="frame",
            leave=True
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

    print(f"\n✅ Extraction complete. Total frames saved: {frame_counter}")


if __name__ == "__main__":
    extract_frames_from_videos(
        video_dir="data/raw_videos",
        output_dir="data/images",
        fps_target=1
    )
