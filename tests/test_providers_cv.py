# System libraries
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import pytest

# Local imports
from mirk.providers_cv import CVProvider


# Create a concrete test class that implements CVProvider
class MockCVProvider(CVProvider):
    """Concrete implementation of CVProvider for testing."""

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)

    def detect_until_object(
        self, source: str, target_class: str, conf_threshold: float = 0.8
    ) -> Optional[Tuple[int, float]]:
        # Implement required abstract method
        return None


class TestCVProvider:
    @pytest.fixture
    def cv_provider(self) -> MockCVProvider:
        """Fixture to create a test provider instance."""
        return MockCVProvider(model_path="dummy_model.pt")

    @pytest.fixture
    def sample_video(self, tmp_path: Path) -> Path:
        """Create a temporary test video file."""
        video_path = tmp_path / "test_video.mp4"

        # Create a simple video with 3 frames
        frames = []
        for i in range(3):
            # Create a frame with different color for each frame
            frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
            frames.append(frame)

        # Write frames to video file
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (100, 100))
        for frame in frames:
            out.write(frame)
        out.release()

        return video_path

    def test_init(self, cv_provider: MockCVProvider) -> None:
        """Test initialization of CVProvider."""
        assert cv_provider.model_path == "dummy_model.pt"

    def test_save_frame_success(
        self,
        cv_provider: MockCVProvider,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        """Test saving a valid frame from video."""
        output_path = tmp_path / "output_frame.jpg"

        # Save frame 1 (second frame)
        cv_provider.save_frame(str(sample_video), 1, str(output_path))

        # Verify the frame was saved
        assert output_path.exists()

        # Read the saved frame and verify its content
        saved_frame = cv2.imread(str(output_path))
        assert saved_frame is not None
        # Increased tolerance to account for potential video compression artifacts
        assert np.mean(saved_frame) == pytest.approx(50, abs=5)

    def test_save_frame_invalid_frame_number(
        self,
        cv_provider: MockCVProvider,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        """Test saving frame with invalid frame number."""
        output_path = tmp_path / "invalid_frame.jpg"

        # Try to save frame beyond video length
        with pytest.raises(ValueError, match="Could not extract frame"):
            cv_provider.save_frame(str(sample_video), 999, str(output_path))

    def test_save_frame_invalid_video_path(
        self, cv_provider: MockCVProvider, tmp_path: Path
    ) -> None:
        """Test saving frame from non-existent video."""
        output_path = tmp_path / "nonexistent_frame.jpg"

        with pytest.raises(ValueError, match="Could not extract frame"):
            cv_provider.save_frame("nonexistent_video.mp4", 0, str(output_path))

    def test_abstract_method_implementation(self) -> None:
        """Test that CVProvider cannot be instantiated without implementing abstract method."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CVProvider("dummy_model.pt")
