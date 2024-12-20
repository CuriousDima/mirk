"""
This pipeline connects to a camera stream and utilizes a specific CV provider to identify a particular object class. 
Upon identification, it forwards an image (a frame) to a VLM provider along with a prompt.

To manage costs and prevent overspending, the pipeline halts after reaching the default maximum number of VLM invocations. 
This limit can be set to infinity if using a local VLM or you understand what you are doing.
"""

import math

from mirk.providers_cv import CVProvider, YOLOProvider
from mirk.providers_vlm import VLMProvider, LlavaAppleSiliconVLMProvider


class StreamingPipeline:
    def __init__(
        self,
        cv_provider: CVProvider,
        vlm_provider: VLMProvider,
        target_class: str,
        conf_threshold: float = 0.5,
        prompt: str = "What is going on in this image?",
        max_invocations: int = 5,
    ):
        """Initialize the streaming pipeline.

        Args:
            cv_provider: CV provider.
            vlm_provider: VLM provider.
            target_class: Target class to detect.
            conf_threshold: Confidence threshold for detection.
            prompt: Prompt for the VLM provider.
            max_invocations: Maximum number of invocations.
        """
        self.cv_provider = cv_provider
        self.vlm_provider = vlm_provider
        self.target_class = target_class
        self.conf_threshold = conf_threshold
        self.prompt = prompt
        self.max_invocations = max_invocations
        self.invocations = 0

    def reason(
        self,
        source: int = 0,
        return_pil: bool = True,
        debug: bool = False,
    ):
        """Reason about the image.

        Args:
            source: Source of the image. Defaults to 0.
            debug: Whether to print debug information. Defaults to False.

        Returns:
            Generator[str, None, None]: Generator of reasons.
        """
        for frame_str in self.cv_provider.detect_stream(
            source,
            self.target_class,
            self.conf_threshold,
            return_pil=return_pil,
            debug=debug,
        ):
            yield self.vlm_provider.ask_about_image(frame_str, self.prompt)
            self.invocations += 1
            if self.invocations >= self.max_invocations:
                return


if __name__ == "__main__":
    cv_provider = YOLOProvider()
    vlm_provider = LlavaAppleSiliconVLMProvider(temperature=0.9)
    pipeline = StreamingPipeline(
        cv_provider,
        vlm_provider,
        target_class="person",
        prompt="USER: <image>\nHow many people are there and what nationality are they?\nASSISTANT:",
        max_invocations=math.inf,
    )
    for reason in pipeline.reason():
        print(reason)
