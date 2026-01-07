"""Provider registry for managing LLM provider implementations.

The registry maintains a mapping of provider names to provider classes,
enabling dynamic provider selection without modifying core logic.
"""

from typing import Dict, List, Type

from .base import BaseProvider


class ProviderRegistry:
    """Registry for LLM provider implementations.
    
    Provides a central location for registering and retrieving provider
    classes by name. This enables the pluggable provider architecture
    where new providers can be added without modifying existing code.
    
    Usage:
        # Register a provider
        ProviderRegistry.register("bedrock", BedrockProvider)
        
        # Get a provider class
        provider_class = ProviderRegistry.get("bedrock")
        provider = provider_class(region="us-east-1")
        
        # List available providers
        providers = ProviderRegistry.list_providers()
    """

    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider implementation.
        
        Args:
            name: The name to register the provider under (e.g., 'bedrock')
            provider_class: The provider class to register
            
        Raises:
            ValueError: If name is empty or provider_class is not a BaseProvider subclass
        """
        if not name or not name.strip():
            raise ValueError("Provider name cannot be empty")
        
        if not isinstance(provider_class, type) or not issubclass(provider_class, BaseProvider):
            raise ValueError(
                f"provider_class must be a subclass of BaseProvider, got {type(provider_class)}"
            )
        
        cls._providers[name] = provider_class

    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Get a provider class by name.
        
        Args:
            name: The name of the provider to retrieve
            
        Returns:
            The provider class registered under that name
            
        Raises:
            ValueError: If the provider name is not registered
        """
        if name not in cls._providers:
            available = ", ".join(sorted(cls._providers.keys())) or "none"
            raise ValueError(f"Unknown provider '{name}'. Available providers: {available}")
        
        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.
        
        Returns:
            Sorted list of registered provider names
        """
        return sorted(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            name: The provider name to check
            
        Returns:
            True if the provider is registered, False otherwise
        """
        return name in cls._providers

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a provider.
        
        Args:
            name: The provider name to unregister
            
        Returns:
            True if the provider was unregistered, False if it wasn't registered
        """
        if name in cls._providers:
            del cls._providers[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers.
        
        Primarily useful for testing to reset state between tests.
        """
        cls._providers.clear()
