from abc import ABC, abstractmethod
from typing import Generator, Tuple
from pathlib import Path
import base64
from PIL import Image
from typing import Union

import cv2
from ultralytics import YOLO


class CVProvider(ABC):
    """Abstract base class for CV providers.

    Provides interface and common functionality for computer vision model providers.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the CV model.

        Args:
            model_path: Path to model weights.
        """
        self.model_path = model_path

    @abstractmethod
    def detect(
        self, source: str, target_class: str, conf_threshold: float = 0.8
    ) -> Generator[Tuple[int, float], None, None]:
        """Run inference on video until specified object is detected.

        Args:
            source: Path to video file.
            target_class: Class name to look for (e.g., 'person', 'car').
            conf_threshold: Confidence threshold for detection (0-1).

        Returns:
            Generator[Tuple[int, float], None, None]: Generator of (frame_number, confidence)
                where object was detected.
        """
        pass

    @abstractmethod
    def detect_stream(
        self,
        source: int,
        target_class: str,
        conf_threshold: float = 0.8,
        return_pil: bool = False,
        debug: bool = False,
    ) -> Generator[Union[str, Image.Image], None, None]:
        """Run inference on camera stream until specified object is detected.

        Args:
            source: Camera index (e.g., 0 for default camera).
            target_class: Class name to look for (e.g., 'person', 'car').
            conf_threshold: Confidence threshold for detection (0-1).
            debug: If True, save the frame to a file.

        Returns:
            Generator[str, None, None]: Generator of
                base64 encoded string of the frame image where object was detected.
        """
        pass

    def get_encoded_frame(self, video_path: str, frame_number: int) -> str:
        """Get a specific frame from a video file as a binary base64 encoded string.

        Args:
            video_path: Path to the video file.
            frame_number: Frame number to get.

        Returns:
            str: Base64 encoded frame image.

        Raises:
            ValueError: If the specified frame cannot be extracted from the video.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not extract frame {frame_number} from video")

        # Encode frame to jpg format
        _, buffer = cv2.imencode(".jpg", frame)
        # Convert to base64 string
        return base64.b64encode(buffer).decode("utf-8")

    def get_frame_as_image(self, video_path: str, frame_number: int) -> Image.Image:
        """Get a specific frame from a video file as a PIL image.

        Args:
            video_path: Path to the video file.
            frame_number: Frame number to get.

        Returns:
            Image.Image: PIL image of the frame.
        """
        pass

    def save_frame(self, video_path: str, frame_number: int, output_path: str) -> None:
        """Save a specific frame from a video file as an image.

        Args:
            video_path: Path to the video file.
            frame_number: Frame number to save.
            output_path: Path where to save the frame image.

        Raises:
            ValueError: If the specified frame cannot be extracted from the video.
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
    """Provider class for YOLO-based computer vision model functionality."""

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

    def detect(
        self, source: str, target_class: str, conf_threshold: float = 0.5
    ) -> Generator[Tuple[int, float], None, None]:
        """Run inference on video until specified object is detected.

        Args:
            source: Path to video file.
            target_class: Class name to look for (e.g., 'person', 'car').
            conf_threshold: Confidence threshold for detection (0-1).

        Returns:
            Generator[Tuple[int, float], None, None]: Generator of (frame_number, confidence)
                where object was detected.
        """
        # self.model(source, stream=True) returns a generator of results.
        # It allows us to pause the inference and resume it later,
        # so we can pass control back here once a frame is processed.
        results = self.model(source, stream=True)
        for frame_idx, result in enumerate(results):
            for current_result in result:
                for box in current_result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    predicted_class = current_result.names[class_id]
                    if predicted_class == target_class and confidence >= conf_threshold:
                        yield frame_idx, confidence

    def detect_stream(
        self,
        source: int,
        target_class: str,
        conf_threshold: float = 0.5,
        return_pil: bool = False,
        debug: bool = False,
    ) -> Generator[Union[str, Image.Image], None, None]:
        """Run inference on camera stream until specified object is detected.

        Args:
            source: Camera index (e.g., 0 for default camera).
            target_class: Class name to look for (e.g., 'person', 'car').
            conf_threshold: Confidence threshold for detection (0-1).
            debug: If True, save the frame to a file.

        Returns:
            Generator[str, None, None]: Generator of
                base64 encoded string of the frame image where object was detected.

        Raises:
            RuntimeError: If camera cannot be opened.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {source}.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Could not read frame from camera.")

                results = self.model.predict(frame, verbose=False)

                if len(results) > 0:
                    for current_result in results[0]:
                        for box in current_result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            predicted_class = current_result.names[class_id]
                            if (
                                predicted_class == target_class
                                and confidence >= conf_threshold
                            ):
                                if return_pil:
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    yield Image.fromarray(rgb_frame)
                                else:
                                    # Encode frame to base64
                                    _, buffer = cv2.imencode(".jpg", frame)
                                    if debug:
                                        cv2.imwrite("output.jpg", frame)
                                    yield base64.b64encode(buffer).decode("utf-8")
        finally:
            cap.release()
