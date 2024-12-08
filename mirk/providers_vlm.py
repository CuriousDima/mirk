from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class VLMProvider(ABC):
    """Abstract base class for Vision-Language Model providers."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the VLM provider.

        Args:
            api_key: API key for the service. If None, will try to get from environment variables.
        """
        self.api_key = api_key

    @abstractmethod
    def ask_about_image(self, image_path: str, question: str) -> str:
        """Ask a question about an image.

        Args:
            image_path: Path to the image file
            question: Question about the image

        Returns:
            str: Model's response to the question
        """
        pass


class OpenAIVLMProvider(VLMProvider):
    """Provider for OpenAI's GPT-4o Vision API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        """Initialize the OpenAI VLM provider.

        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
            model: Model to use. Defaults to GPT-4 Vision.
        """
        super().__init__(api_key)
        # Use provided api_key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file or pass it directly."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def ask_about_image(self, image_path: str, question: str) -> str:
        """Ask a question about an image using GPT-4o.

        Args:
            image_path: Path to the image file
            question: Question about the image

        Returns:
            str: Model's response to the question

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image path is invalid
        """
        # Verify image path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Encode image
        base64_image = self._encode_image(str(image_path))

        # Prepare the messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        # Make the API call
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=300
        )

        return response.choices[0].message.content
