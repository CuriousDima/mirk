# System libraries
import pytest

# Local imports
from mirk.providers_vlm import VLMProvider


class TestVLMProvider:
    """Test suite for VLMProvider class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that VLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            VLMProvider()

    def test_concrete_subclass_must_implement_abstract_method(self):
        """Test that concrete subclasses must implement ask_about_image."""

        class IncompleteProvider(VLMProvider):
            pass

        with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
            IncompleteProvider()

    def test_can_create_concrete_subclass(self):
        """Test that we can create a concrete subclass with all methods implemented."""

        class ConcreteProvider(VLMProvider):
            def ask_about_image(self, image_path: str, question: str) -> str:
                return "test response"

        provider = ConcreteProvider()
        assert isinstance(provider, VLMProvider)

    def test_api_key_initialization(self):
        """Test that api_key is properly set during initialization."""

        class ConcreteProvider(VLMProvider):
            def ask_about_image(self, image_path: str, question: str) -> str:
                return "test response"

        # Test with explicit api_key
        provider = ConcreteProvider(api_key="test_key")
        assert provider.api_key == "test_key"

        # Test with default None
        provider = ConcreteProvider()
        assert provider.api_key is None
