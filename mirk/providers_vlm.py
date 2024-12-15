from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class VLMProvider(ABC):
    """Abstract base class for Vision-Language Model providers.

    Provides interface and common functionality for vision-language models.
    """

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
            image_path: Path to the image file.
            question: Question about the image.

        Returns:
            str: Model's response to the question.
        """
        pass


class OpenAIVLMProvider(VLMProvider):
    """Provider for OpenAI's GPT-4 Vision API."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "gpt-4-vision-preview"
    ) -> None:
        """Initialize the OpenAI VLM provider.

        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
            model: Model to use. Defaults to GPT-4 Vision.

        Raises:
            ValueError: If OpenAI API key is not provided and not found in environment.
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

    def ask_about_image(self, base64_image: str, question: str) -> str:
        """Ask a question about an image using GPT-4 Vision.

        Args:
            base64_image: Base64 encoded image.
            question: Question about the image.

        Returns:
            str: Model's response to the question.

        """
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
