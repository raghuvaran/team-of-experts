"""Property-based tests for TOXP provider architecture.

Feature: toxp-cli
"""

from typing import AsyncIterator

import pytest
from hypothesis import given, strategies as st, settings, assume
from pydantic import ValidationError

from toxp.providers import BaseProvider, ProviderResponse, ProviderRegistry


# Create a concrete test provider for testing the registry
class MockProvider(BaseProvider):
    """Mock provider for testing purposes."""

    def __init__(self, model_id: str = "test-model"):
        self._model_id = model_id

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return self._model_id

    async def invoke_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        return ProviderResponse(
            text="Mock response",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            model_id=self._model_id,
        )

    async def invoke_model_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        yield "Mock"
        yield " response"


# Strategy for generating valid provider names
valid_provider_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() == x and len(x) > 0)


class TestProviderRegistryRoundTrip:
    """Property tests for provider registry round-trip.
    
    Property 5: Provider Registry Round-Trip
    Validates: Requirements 4.2, 4.3
    
    For any provider class registered with a name, retrieving the provider
    by that name SHALL return the same class that was registered.
    """

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry.clear()

    @given(name=valid_provider_names)
    @settings(max_examples=100)
    def test_register_then_get_returns_same_class(self, name: str) -> None:
        """Property: Registering a provider then getting it returns the same class.
        
        Property 5: Provider Registry Round-Trip
        Validates: Requirements 4.2, 4.3
        """
        # Clear registry for each test iteration
        ProviderRegistry.clear()
        
        # Register the provider
        ProviderRegistry.register(name, MockProvider)
        
        # Get the provider
        retrieved = ProviderRegistry.get(name)
        
        # Should be the exact same class
        assert retrieved is MockProvider

    @given(name=valid_provider_names)
    @settings(max_examples=100)
    def test_registered_provider_appears_in_list(self, name: str) -> None:
        """Property: A registered provider appears in list_providers.
        
        Property 5: Provider Registry Round-Trip
        Validates: Requirements 4.2, 4.3
        """
        ProviderRegistry.clear()
        
        # Register the provider
        ProviderRegistry.register(name, MockProvider)
        
        # Should appear in list
        providers = ProviderRegistry.list_providers()
        assert name in providers

    @given(names=st.lists(valid_provider_names, min_size=1, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_multiple_providers_all_retrievable(self, names: list) -> None:
        """Property: Multiple registered providers are all retrievable.
        
        Property 5: Provider Registry Round-Trip
        Validates: Requirements 4.2, 4.3
        """
        ProviderRegistry.clear()
        
        # Create unique mock provider classes for each name
        provider_classes = {}
        for name in names:
            # Create a unique class for each name
            provider_class = type(f"MockProvider_{name}", (MockProvider,), {})
            provider_classes[name] = provider_class
            ProviderRegistry.register(name, provider_class)
        
        # All should be retrievable
        for name, expected_class in provider_classes.items():
            retrieved = ProviderRegistry.get(name)
            assert retrieved is expected_class

    @given(name=valid_provider_names)
    @settings(max_examples=100)
    def test_is_registered_returns_true_after_registration(self, name: str) -> None:
        """Property: is_registered returns True for registered providers.
        
        Property 5: Provider Registry Round-Trip
        Validates: Requirements 4.2, 4.3
        """
        ProviderRegistry.clear()
        
        # Not registered yet
        assert not ProviderRegistry.is_registered(name)
        
        # Register
        ProviderRegistry.register(name, MockProvider)
        
        # Now registered
        assert ProviderRegistry.is_registered(name)

    @given(name=valid_provider_names)
    @settings(max_examples=100)
    def test_unregistered_provider_raises_error(self, name: str) -> None:
        """Property: Getting an unregistered provider raises ValueError.
        
        Property 5: Provider Registry Round-Trip
        Validates: Requirements 4.2, 4.3
        """
        ProviderRegistry.clear()
        
        # Should raise ValueError for unregistered provider
        with pytest.raises(ValueError) as exc_info:
            ProviderRegistry.get(name)
        
        assert f"Unknown provider '{name}'" in str(exc_info.value)


class TestProviderResponseStructure:
    """Property tests for provider response structure.
    
    Property 6: Provider Response Structure
    Validates: Requirements 4.7
    
    For any successful provider invocation, the response SHALL contain:
    text (non-empty string), input_tokens (non-negative integer),
    output_tokens (non-negative integer), latency_ms (positive float),
    and model_id (non-empty string).
    """

    @given(
        text=st.text(min_size=1).filter(lambda x: x.strip()),
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
        latency_ms=st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        model_id=st.text(min_size=1).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_valid_response_structure(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_id: str,
    ) -> None:
        """Property: Valid inputs create a valid ProviderResponse with all required fields.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        response = ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=model_id,
        )
        
        # Verify all fields are present and have correct types
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        
        assert isinstance(response.input_tokens, int)
        assert response.input_tokens >= 0
        
        assert isinstance(response.output_tokens, int)
        assert response.output_tokens >= 0
        
        assert isinstance(response.latency_ms, float)
        assert response.latency_ms > 0
        
        assert isinstance(response.model_id, str)
        assert len(response.model_id) > 0

    @given(
        text=st.text(min_size=1).filter(lambda x: x.strip()),
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
        latency_ms=st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        model_id=st.text(min_size=1).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_response_round_trip_via_dict(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_id: str,
    ) -> None:
        """Property: ProviderResponse round-trips through dict serialization.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        original = ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=model_id,
        )
        
        # Round-trip through dict
        as_dict = original.to_dict()
        restored = ProviderResponse.from_dict(as_dict)
        
        assert restored.text == original.text
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert abs(restored.latency_ms - original.latency_ms) < 1e-10
        assert restored.model_id == original.model_id

    @given(
        text=st.text(min_size=1).filter(lambda x: x.strip()),
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
        latency_ms=st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        model_id=st.text(min_size=1).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_response_round_trip_via_json(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_id: str,
    ) -> None:
        """Property: ProviderResponse round-trips through JSON serialization.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        original = ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=model_id,
        )
        
        # Round-trip through JSON
        as_json = original.to_json()
        restored = ProviderResponse.from_json(as_json)
        
        assert restored.text == original.text
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert abs(restored.latency_ms - original.latency_ms) < 1e-10
        assert restored.model_id == original.model_id

    @given(
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
        latency_ms=st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        model_id=st.text(min_size=1).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_empty_text_rejected(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_id: str,
    ) -> None:
        """Property: Empty text is rejected with ValidationError.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        with pytest.raises(ValidationError):
            ProviderResponse(
                text="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model_id=model_id,
            )

    @given(
        text=st.text(min_size=1).filter(lambda x: x.strip()),
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
        latency_ms=st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_empty_model_id_rejected(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Property: Empty model_id is rejected with ValidationError.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        with pytest.raises(ValidationError):
            ProviderResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model_id="",
            )

    @given(
        text=st.text(min_size=1).filter(lambda x: x.strip()),
        negative_tokens=st.integers(max_value=-1),
        output_tokens=st.integers(min_value=0, max_value=1000000),
        latency_ms=st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        model_id=st.text(min_size=1).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_negative_input_tokens_rejected(
        self,
        text: str,
        negative_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_id: str,
    ) -> None:
        """Property: Negative input_tokens is rejected with ValidationError.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        with pytest.raises(ValidationError):
            ProviderResponse(
                text=text,
                input_tokens=negative_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model_id=model_id,
            )

    @given(
        text=st.text(min_size=1).filter(lambda x: x.strip()),
        input_tokens=st.integers(min_value=0, max_value=1000000),
        negative_tokens=st.integers(max_value=-1),
        latency_ms=st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        model_id=st.text(min_size=1).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_negative_output_tokens_rejected(
        self,
        text: str,
        input_tokens: int,
        negative_tokens: int,
        latency_ms: float,
        model_id: str,
    ) -> None:
        """Property: Negative output_tokens is rejected with ValidationError.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        with pytest.raises(ValidationError):
            ProviderResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=negative_tokens,
                latency_ms=latency_ms,
                model_id=model_id,
            )

    @given(
        text=st.text(min_size=1).filter(lambda x: x.strip()),
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
        non_positive_latency=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        model_id=st.text(min_size=1).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_non_positive_latency_rejected(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        non_positive_latency: float,
        model_id: str,
    ) -> None:
        """Property: Non-positive latency_ms is rejected with ValidationError.
        
        Property 6: Provider Response Structure
        Validates: Requirements 4.7
        """
        with pytest.raises(ValidationError):
            ProviderResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=non_positive_latency,
                model_id=model_id,
            )


# Import BedrockProvider for model ID validation tests
from toxp.providers.bedrock import BedrockProvider


# Strategy for generating valid Claude model IDs
valid_claude_model_ids = st.sampled_from([
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-instant-v1",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "eu.anthropic.claude-3-sonnet-20240229-v1:0",
    "ap.anthropic.claude-3-haiku-20240307-v1:0",
    "global.anthropic.claude-3-opus-20240229-v1:0",
])

# Strategy for generating invalid model IDs
invalid_model_ids = st.one_of(
    # Empty strings
    st.just(""),
    # Whitespace only
    st.text(alphabet=" \t\n", min_size=1, max_size=10),
    # Missing anthropic prefix
    st.text(min_size=1, max_size=50).filter(
        lambda x: "anthropic" not in x.lower() and x.strip()
    ),
    # Wrong format patterns
    st.sampled_from([
        "openai.gpt-4",
        "claude-3-sonnet",  # Missing anthropic prefix
        "anthropic-claude-3-sonnet",  # Wrong separator
        "anthropic.gpt-4",  # Wrong model family
        "us.openai.gpt-4",  # Wrong provider
        "invalid.anthropic.claude-3-sonnet",  # Invalid region prefix
        "anthropic.claude",  # Incomplete model name
    ]),
)


class TestModelIdValidation:
    """Property tests for model ID validation.
    
    Property 7: Model ID Validation
    Validates: Requirements 5.7
    
    For any string input to model ID validation:
    - Valid Claude model ID patterns SHALL pass validation
    - Invalid patterns SHALL fail validation with a descriptive error
    """

    @given(model_id=valid_claude_model_ids)
    @settings(max_examples=100)
    def test_valid_model_ids_pass_validation(self, model_id: str) -> None:
        """Property: Valid Claude model IDs pass validation.
        
        Property 7: Model ID Validation
        Validates: Requirements 5.7
        """
        # Should not raise
        result = BedrockProvider.validate_model_id(model_id)
        assert result is True

    @given(model_id=valid_claude_model_ids)
    @settings(max_examples=100)
    def test_is_valid_model_id_returns_true_for_valid(self, model_id: str) -> None:
        """Property: is_valid_model_id returns True for valid model IDs.
        
        Property 7: Model ID Validation
        Validates: Requirements 5.7
        """
        assert BedrockProvider.is_valid_model_id(model_id) is True

    @given(model_id=invalid_model_ids)
    @settings(max_examples=100)
    def test_invalid_model_ids_fail_validation(self, model_id: str) -> None:
        """Property: Invalid model IDs fail validation with descriptive error.
        
        Property 7: Model ID Validation
        Validates: Requirements 5.7
        """
        with pytest.raises(ValueError) as exc_info:
            BedrockProvider.validate_model_id(model_id)
        
        # Error message should be descriptive
        error_msg = str(exc_info.value).lower()
        assert "model id" in error_msg or "model_id" in error_msg

    @given(model_id=invalid_model_ids)
    @settings(max_examples=100)
    def test_is_valid_model_id_returns_false_for_invalid(self, model_id: str) -> None:
        """Property: is_valid_model_id returns False for invalid model IDs.
        
        Property 7: Model ID Validation
        Validates: Requirements 5.7
        """
        assert BedrockProvider.is_valid_model_id(model_id) is False

    def test_none_model_id_fails_validation(self) -> None:
        """Edge case: None model ID fails validation.
        
        Property 7: Model ID Validation
        Validates: Requirements 5.7
        """
        with pytest.raises(ValueError):
            BedrockProvider.validate_model_id(None)  # type: ignore
        
        assert BedrockProvider.is_valid_model_id(None) is False  # type: ignore

    @given(
        region=st.sampled_from(["us", "eu", "ap", "global"]),
        variant=st.sampled_from(["3-sonnet", "3-haiku", "3-opus", "3-5-sonnet", "instant"]),
        version=st.text(alphabet="0123456789-:v", min_size=1, max_size=20),
    )
    @settings(max_examples=100)
    def test_constructed_model_ids_with_valid_pattern(
        self, region: str, variant: str, version: str
    ) -> None:
        """Property: Constructed model IDs with valid pattern pass validation.
        
        Property 7: Model ID Validation
        Validates: Requirements 5.7
        """
        # Construct a model ID with valid pattern
        model_id = f"{region}.anthropic.claude-{variant}-{version}"
        
        # Should pass validation (pattern matches)
        result = BedrockProvider.is_valid_model_id(model_id)
        assert result is True
