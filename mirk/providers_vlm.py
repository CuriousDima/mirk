# Standard library imports
from abc import ABC, abstractmethod
import codecs
import os
from pathlib import Path
from typing import Optional

# Third-party imports
from dotenv import load_dotenv
import mlx.core as mx
from openai import OpenAI
from PIL import Image
import requests
from transformers import AutoProcessor

# Local imports
from mirk.apple_silicon.llava.llava import LlavaModel

# Load environment variables from .env file
load_dotenv()


class VLMProvider(ABC):
    """Abstract base class for Vision-Language Model providers.

    Provides interface and common functionality for vision-language models.
    """

    def __init__(
        self, api_key: Optional[str] = None, model: Optional[str] = None
    ) -> None:
        """Initialize the VLM provider.

        Args:
            api_key: API key for the service. If None, will try to get from environment variables.
            model: VLM model to use. Defaults to None.
        """
        self.api_key = api_key
        self.model = model

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
        self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"
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


class LlavaAppleSiliconVLMProvider(VLMProvider):
    """
    A provider for Apple Silicon using LLaVA-like (Llama-based) inference for VLM tasks.
    """

    def __init__(
        self,
        model: str = "llava-hf/llava-1.5-7b-hf",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        super().__init__(model=model)
        self.processor = AutoProcessor.from_pretrained(model, tokenizer_config={})
        self.model_obj = LlavaModel.from_pretrained(model)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def sample(self, logits: mx.array) -> mx.array:
        """Sample from logits using temperature-based categorical sampling.

        Args:
            logits: Input logits tensor.

        Returns:
            mx.array: Sampled token indices.
        """
        if self.temperature == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / self.temperature))

    def ask_about_image(self, image_obj: Image, question: str) -> str:
        """Ask a question about an image using LLaVA model.

        Args:
            image_obj: PIL Image object to analyze.
            question: Question about the image.

        Returns:
            str: Model's response to the question.
        """
        prompt = codecs.decode(question, "unicode_escape")
        inputs = self.processor(image_obj, prompt, return_tensors="np")
        pixel_values = mx.array(inputs["pixel_values"])
        input_ids = mx.array(inputs["input_ids"])

        logits, cache = self.model_obj(input_ids, pixel_values)
        logits = logits[:, -1, :]
        y = self.sample(logits)
        tokens = [y.item()]

        for n in range(self.max_tokens - 1):
            logits, cache = self.model_obj.language_model(y[None], cache=cache)
            logits = logits[:, -1, :]
            y = self.sample(logits)
            token = y.item()
            if token == self.processor.tokenizer.eos_token_id:
                break
            tokens.append(token)

        return self.processor.tokenizer.decode(tokens)


def load_image(image_source: str) -> Image.Image:
    """Load an image from a URL or local file path.

    Args:
        image_source: URL or file path to the image.

    Returns:
        PIL.Image: Loaded image object.

    Raises:
        ValueError: If the image cannot be loaded from the given source.
    """
    if image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            )
    elif Path(image_source).is_file():
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image {image_source} with error: {e}")
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )


def main() -> None:
    """Run a demo of the LLaVA VLM provider with a sample image."""
    img = load_image(
        "https://hips.hearstapps.com/hmg-prod/images/dachshunds-outside-65532f91a71ff.jpg"
    )
    question = "USER: <image>\nWhat are these?\nASSISTANT:"
    provider = LlavaAppleSiliconVLMProvider()
    print(provider.ask_about_image(img, question))


if __name__ == "__main__":
    main()
