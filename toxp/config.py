"""Configuration management for TOXP CLI.

Handles persistent configuration at ~/.toxp/config.json with support for:
- Environment variable overrides (TOXP_AWS_PROFILE)
- CLI argument overrides
- Configuration precedence: CLI > ENV > file > defaults
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# Configuration directory and file paths
CONFIG_DIR = Path.home() / ".toxp"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Valid configuration keys
VALID_CONFIG_KEYS = (
    "provider",
    "aws_profile",
    "region",
    "model",
    "num_agents",
    "temperature",
    "coordinator_temperature",
    "max_tokens",
    "max_concurrency",
    "log_enabled",
    "log_retention_days",
)

# CLI key mapping (kebab-case to snake_case)
CLI_KEY_MAP = {
    "aws-profile": "aws_profile",
    "num-agents": "num_agents",
    "coordinator-temperature": "coordinator_temperature",
    "max-tokens": "max_tokens",
    "max-concurrency": "max_concurrency",
    "log-enabled": "log_enabled",
    "log-retention-days": "log_retention_days",
}

# Environment variable prefix
ENV_PREFIX = "TOXP_"


class ToxpConfig(BaseModel):
    """User configuration with defaults.
    
    All configuration values have sensible defaults and can be overridden
    via config file, environment variables, or CLI arguments.
    """

    provider: str = Field(
        default="bedrock",
        description="LLM provider to use",
    )
    aws_profile: str = Field(
        default="default",
        description="AWS profile for credentials",
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock",
    )
    model: str = Field(
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        description="Model ID to use",
    )
    num_agents: int = Field(
        default=16,
        ge=2,
        le=32,
        description="Number of reasoning agents (2-32)",
    )
    temperature: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Temperature for reasoning agents",
    )
    coordinator_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for coordinator agent",
    )
    max_tokens: int = Field(
        default=8192,
        ge=1,
        description="Maximum tokens per response",
    )
    max_concurrency: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Maximum concurrent API requests (None = auto-calculate based on model quotas)",
    )
    log_enabled: bool = Field(
        default=True,
        description="Enable session logging",
    )
    log_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days to retain session logs",
    )

    @field_validator("num_agents")
    @classmethod
    def validate_num_agents(cls, v: int) -> int:
        """Validate num_agents is within allowed range."""
        if not 2 <= v <= 32:
            raise ValueError("num_agents must be between 2 and 32")
        return v

    @field_validator("temperature", "coordinator_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within allowed range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToxpConfig":
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ToxpConfig":
        """Create from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def get_defaults(cls) -> "ToxpConfig":
        """Get default configuration."""
        return cls()


def _normalize_key(key: str) -> str:
    """Normalize a config key from CLI format (kebab-case) to internal format (snake_case)."""
    return CLI_KEY_MAP.get(key, key.replace("-", "_"))


def _denormalize_key(key: str) -> str:
    """Convert internal key (snake_case) to CLI format (kebab-case)."""
    return key.replace("_", "-")


class ConfigManager:
    """Manages persistent configuration at ~/.toxp/config.json.
    
    Supports configuration precedence: CLI args > environment variables > config file > defaults.
    """

    def __init__(self, config_dir: Optional[Path] = None, config_file: Optional[Path] = None):
        """Initialize ConfigManager with optional custom paths (for testing)."""
        self._config_dir = config_dir or CONFIG_DIR
        self._config_file = config_file or (self._config_dir / "config.json")

    @property
    def config_path(self) -> Path:
        """Get the configuration file path."""
        return self._config_file

    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from file, returning empty dict if not found."""
        if not self._config_file.exists():
            return {}
        try:
            with open(self._config_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_to_file(self, data: Dict[str, Any]) -> None:
        """Save configuration to file."""
        self._ensure_config_dir()
        with open(self._config_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables.
        
        Supports TOXP_AWS_PROFILE and other TOXP_* environment variables.
        """
        overrides: Dict[str, Any] = {}
        
        # Check for TOXP_AWS_PROFILE specifically (as per requirements)
        aws_profile = os.environ.get("TOXP_AWS_PROFILE")
        if aws_profile is not None:
            overrides["aws_profile"] = aws_profile
        
        # Check for other TOXP_* environment variables
        for key in VALID_CONFIG_KEYS:
            env_key = f"{ENV_PREFIX}{key.upper()}"
            env_value = os.environ.get(env_key)
            if env_value is not None and key not in overrides:
                overrides[key] = self._parse_env_value(key, env_value)
        
        return overrides

    def _parse_env_value(self, key: str, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Get the field type from ToxpConfig
        defaults = ToxpConfig.get_defaults()
        default_value = getattr(defaults, key)
        
        if isinstance(default_value, bool):
            return value.lower() in ("true", "1", "yes", "on")
        elif isinstance(default_value, int):
            return int(value)
        elif isinstance(default_value, float):
            return float(value)
        return value

    def load(self) -> ToxpConfig:
        """Load configuration with precedence: ENV > file > defaults.
        
        Note: CLI args are applied separately via apply_overrides().
        """
        # Start with defaults
        config_data = ToxpConfig.get_defaults().to_dict()
        
        # Apply file values
        file_data = self._load_from_file()
        for key, value in file_data.items():
            if key in VALID_CONFIG_KEYS:
                config_data[key] = value
        
        # Apply environment variable overrides
        env_overrides = self._get_env_overrides()
        for key, value in env_overrides.items():
            config_data[key] = value
        
        # Create config on first use if file doesn't exist
        if not self._config_file.exists():
            self._save_to_file(config_data)
        
        return ToxpConfig.from_dict(config_data)

    def save(self, config: ToxpConfig) -> None:
        """Save configuration to file."""
        self._save_to_file(config.to_dict())

    def get(self, key: str) -> Any:
        """Get a single configuration value.
        
        Args:
            key: Configuration key (supports both kebab-case and snake_case)
            
        Returns:
            The configuration value
            
        Raises:
            KeyError: If the key is not a valid configuration key
        """
        normalized_key = _normalize_key(key)
        if normalized_key not in VALID_CONFIG_KEYS:
            raise KeyError(f"Unknown configuration key: {key}")
        
        config = self.load()
        return getattr(config, normalized_key)

    def set(self, key: str, value: Any) -> None:
        """Set a single configuration value.
        
        Args:
            key: Configuration key (supports both kebab-case and snake_case)
            value: Value to set (will be converted to appropriate type)
            
        Raises:
            KeyError: If the key is not a valid configuration key
            ValueError: If the value is invalid for the key
        """
        normalized_key = _normalize_key(key)
        if normalized_key not in VALID_CONFIG_KEYS:
            raise KeyError(f"Unknown configuration key: {key}")
        
        # Load current config from file only (not env vars) to preserve user settings
        file_data = self._load_from_file()
        if not file_data:
            file_data = ToxpConfig.get_defaults().to_dict()
        
        # Convert value to appropriate type
        converted_value = self._convert_value(normalized_key, value)
        
        # Validate by creating a test config
        test_data = file_data.copy()
        test_data[normalized_key] = converted_value
        ToxpConfig.from_dict(test_data)  # This will raise if invalid
        
        # Save to file
        file_data[normalized_key] = converted_value
        self._save_to_file(file_data)

    def _convert_value(self, key: str, value: Any) -> Any:
        """Convert a value to the appropriate type for a key.
        
        Special handling:
        - "auto" for max_concurrency resets to None (auto-calculate)
        """
        # Handle "auto" keyword for max_concurrency
        if key == "max_concurrency" and isinstance(value, str) and value.lower() == "auto":
            return None
        
        if isinstance(value, str):
            defaults = ToxpConfig.get_defaults()
            default_value = getattr(defaults, key)
            
            if isinstance(default_value, bool):
                return value.lower() in ("true", "1", "yes", "on")
            elif isinstance(default_value, int):
                return int(value)
            elif isinstance(default_value, float):
                return float(value)
        return value

    def reset(self) -> None:
        """Reset configuration to defaults."""
        defaults = ToxpConfig.get_defaults()
        self._save_to_file(defaults.to_dict())

    def show(self) -> Dict[str, Any]:
        """Get all current configuration values.
        
        Returns:
            Dictionary of all configuration key-value pairs
        """
        config = self.load()
        return config.to_dict()

    def apply_overrides(self, config: ToxpConfig, cli_args: Dict[str, Any]) -> ToxpConfig:
        """Apply CLI argument overrides to configuration.
        
        This implements the highest precedence level (CLI > ENV > file > defaults).
        
        Args:
            config: Base configuration (already has ENV and file values)
            cli_args: CLI argument overrides (may use kebab-case keys)
            
        Returns:
            New ToxpConfig with CLI overrides applied
        """
        config_data = config.to_dict()
        
        for key, value in cli_args.items():
            if value is not None:
                normalized_key = _normalize_key(key)
                if normalized_key in VALID_CONFIG_KEYS:
                    config_data[normalized_key] = value
        
        return ToxpConfig.from_dict(config_data)

    def get_valid_keys(self) -> List[str]:
        """Get list of valid configuration keys in CLI format (kebab-case)."""
        return [_denormalize_key(k) for k in VALID_CONFIG_KEYS]
