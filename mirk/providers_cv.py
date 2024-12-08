from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path

import cv2
from ultralytics import YOLO


class CVProvider(ABC):
    """Abstract base class for CV providers."""

    def __init__(self, model_path: str) -> None:
        """Initialize the CV model.

        Args:
            model_path: Path to model weights.
        """
        self.model_path = model_path

    @abstractmethod
    def detect_until_object(
        self, source: str, target_class: str, conf_threshold: float = 0.8
    ) -> Optional[Tuple[int, float]]:
        """Run inference on video until specified object is detected.

        Args:
            source: Path to video file
            target_class: Class name to look for (e.g., 'person', 'car')
            conf_threshold: Confidence threshold for detection (0-1)

        Returns:
            tuple: (frame_number, confidence) where object was detected, or None if not found
        """
        pass

    def save_frame(self, video_path: str, frame_number: int, output_path: str) -> None:
        """Save a specific frame from a video file as an image.

        Args:
            video_path: Path to the video file
            frame_number: Frame number to save
            output_path: Path where to save the frame image
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(output_path, frame)
        else:
            raise ValueError(f"Could not extract frame {frame_number} from video")

        cap.release()


class YOLOProvider(CVProvider):
    """A provider class for YOLO-based computer vision model functionality."""

    def __init__(self, model_path: str = "yolo11n.pt"):
        """Initialize YOLO model.

        Args:
            model_path: Path to YOLO model weights. Defaults to YOLOv11 nano model.
        """
        # Create models directory if it doesn't exist
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(models_dir / model_path)
        self.model = YOLO(self.model_path)

    def detect_until_object(
        self, source: str, target_class: str, conf_threshold: float = 0.5
    ) -> Optional[Tuple[int, float]]:
        """Run inference on video until specified object is detected.

        Args:
            source: Path to video file
            target_class: Class name to look for (e.g., 'person', 'car')
            conf_threshold: Confidence threshold for detection (0-1)

        Returns:
            tuple: (frame_number, confidence) where object was detected, or None if not found
        """
        results = self.model(source, stream=True)  # Enable streaming for video

        for frame_idx, result in enumerate(results):
            # Check detections in current frame
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                predicted_class = result.names[class_id]
                if predicted_class == target_class and confidence >= conf_threshold:
                    return frame_idx, confidence

        return None  # Object not found
