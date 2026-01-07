"""Property-based tests for TOXP configuration management.

Feature: toxp-cli
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings, assume

from toxp.config import (
    ToxpConfig,
    ConfigManager,
    VALID_CONFIG_KEYS,
    _normalize_key,
    _denormalize_key,
)


# Strategies for generating valid configuration values
valid_providers = st.sampled_from(["bedrock", "anthropic", "openai", "custom"])
valid_aws_profiles = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() == x and len(x) > 0)
valid_regions = st.sampled_from([
    "us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1", "us-east-2"
])
valid_models = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters=".-_:"),
    min_size=1,
    max_size=100,
).filter(lambda x: x.strip() == x and len(x) > 0)
valid_num_agents = st.integers(min_value=2, max_value=32)
valid_temperatures = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_max_tokens = st.integers(min_value=1, max_value=100000)
valid_log_enabled = st.booleans()
valid_log_retention_days = st.integers(min_value=1, max_value=365)


class TestConfigRoundTrip:
    """Property tests for configuration round-trip consistency.
    
    Property 1: Configuration Round-Trip Consistency
    Validates: Requirements 2.2, 2.3
    
    For any valid configuration key and value, setting the value via `config set`
    then retrieving it via `config get` SHALL return the same value.
    """

    @given(provider=valid_providers)
    @settings(max_examples=100)
    def test_provider_round_trip(self, provider: str) -> None:
        """Property: provider value round-trips through set/get.
        
        Property 1: Configuration Round-Trip Consistency
        Validates: Requirements 2.2, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            manager.set("provider", provider)
            retrieved = manager.get("provider")
            
            assert retrieved == provider

    @given(aws_profile=valid_aws_profiles)
    @settings(max_examples=100)
    def test_aws_profile_round_trip(self, aws_profile: str) -> None:
        """Property: aws-profile value round-trips through set/get.
        
        Property 1: Configuration Round-Trip Consistency
        Validates: Requirements 2.2, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            # Test with kebab-case key
            manager.set("aws-profile", aws_profile)
            retrieved = manager.get("aws-profile")
            
            assert retrieved == aws_profile

    @given(region=valid_regions)
    @settings(max_examples=100)
    def test_region_round_trip(self, region: str) -> None:
        """Property: region value round-trips through set/get.
        
        Property 1: Configuration Round-Trip Consistency
        Validates: Requirements 2.2, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            manager.set("region", region)
            retrieved = manager.get("region")
            
            assert retrieved == region

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_num_agents_round_trip(self, num_agents: int) -> None:
        """Property: num-agents value round-trips through set/get.
        
        Property 1: Configuration Round-Trip Consistency
        Validates: Requirements 2.2, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            # Test with string value (as would come from CLI)
            manager.set("num-agents", str(num_agents))
            retrieved = manager.get("num-agents")
            
            assert retrieved == num_agents

    @given(temperature=valid_temperatures)
    @settings(max_examples=100)
    def test_temperature_round_trip(self, temperature: float) -> None:
        """Property: temperature value round-trips through set/get.
        
        Property 1: Configuration Round-Trip Consistency
        Validates: Requirements 2.2, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            manager.set("temperature", str(temperature))
            retrieved = manager.get("temperature")
            
            # Float comparison with tolerance for JSON serialization
            assert abs(retrieved - temperature) < 1e-10

    @given(log_enabled=valid_log_enabled)
    @settings(max_examples=100)
    def test_log_enabled_round_trip(self, log_enabled: bool) -> None:
        """Property: log-enabled value round-trips through set/get.
        
        Property 1: Configuration Round-Trip Consistency
        Validates: Requirements 2.2, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            # Test with string value (as would come from CLI)
            manager.set("log-enabled", str(log_enabled).lower())
            retrieved = manager.get("log-enabled")
            
            assert retrieved == log_enabled

    @given(log_retention_days=valid_log_retention_days)
    @settings(max_examples=100)
    def test_log_retention_days_round_trip(self, log_retention_days: int) -> None:
        """Property: log-retention-days value round-trips through set/get.
        
        Property 1: Configuration Round-Trip Consistency
        Validates: Requirements 2.2, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            manager.set("log-retention-days", str(log_retention_days))
            retrieved = manager.get("log-retention-days")
            
            assert retrieved == log_retention_days



class TestConfigPrecedence:
    """Property tests for configuration precedence.
    
    Property 4: Configuration Precedence
    Validates: Requirements 2.9
    
    For any configuration key with values set at multiple levels (CLI arg, environment
    variable, config file, default), the effective value SHALL follow the precedence:
    CLI args > environment variables > config file > defaults.
    """

    @given(
        file_value=valid_aws_profiles,
        env_value=valid_aws_profiles,
        cli_value=valid_aws_profiles,
    )
    @settings(max_examples=100)
    def test_cli_overrides_env_and_file(
        self, file_value: str, env_value: str, cli_value: str
    ) -> None:
        """Property: CLI args have highest precedence over ENV and file values.
        
        Property 4: Configuration Precedence
        Validates: Requirements 2.9
        """
        import os
        
        # Ensure all values are distinct for meaningful test
        assume(file_value != env_value and env_value != cli_value and file_value != cli_value)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            # Set file value
            manager.set("aws-profile", file_value)
            
            # Set env value
            old_env = os.environ.get("TOXP_AWS_PROFILE")
            try:
                os.environ["TOXP_AWS_PROFILE"] = env_value
                
                # Load config (should have env value due to ENV > file)
                config = manager.load()
                assert config.aws_profile == env_value, "ENV should override file"
                
                # Apply CLI override (CLI > ENV > file)
                config = manager.apply_overrides(config, {"aws-profile": cli_value})
                assert config.aws_profile == cli_value, "CLI should override ENV"
                
            finally:
                if old_env is None:
                    os.environ.pop("TOXP_AWS_PROFILE", None)
                else:
                    os.environ["TOXP_AWS_PROFILE"] = old_env

    @given(
        file_value=valid_aws_profiles,
        env_value=valid_aws_profiles,
    )
    @settings(max_examples=100)
    def test_env_overrides_file(self, file_value: str, env_value: str) -> None:
        """Property: Environment variables override config file values.
        
        Property 4: Configuration Precedence
        Validates: Requirements 2.9
        """
        import os
        
        assume(file_value != env_value)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
            
            # Set file value
            manager.set("aws-profile", file_value)
            
            # Set env value
            old_env = os.environ.get("TOXP_AWS_PROFILE")
            try:
                os.environ["TOXP_AWS_PROFILE"] = env_value
                
                # Load config (should have env value)
                config = manager.load()
                assert config.aws_profile == env_value, "ENV should override file"
                
            finally:
                if old_env is None:
                    os.environ.pop("TOXP_AWS_PROFILE", None)
                else:
                    os.environ["TOXP_AWS_PROFILE"] = old_env

    @given(file_value=valid_aws_profiles)
    @settings(max_examples=100)
    def test_file_overrides_defaults(self, file_value: str) -> None:
        """Property: Config file values override defaults.
        
        Property 4: Configuration Precedence
        Validates: Requirements 2.9
        """
        import os
        
        default_value = ToxpConfig.get_defaults().aws_profile
        assume(file_value != default_value)
        
        # Ensure no env var interference
        old_env = os.environ.get("TOXP_AWS_PROFILE")
        try:
            os.environ.pop("TOXP_AWS_PROFILE", None)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config_dir = Path(tmpdir)
                manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
                
                # Set file value
                manager.set("aws-profile", file_value)
                
                # Load config (should have file value)
                config = manager.load()
                assert config.aws_profile == file_value, "File should override defaults"
                
        finally:
            if old_env is not None:
                os.environ["TOXP_AWS_PROFILE"] = old_env

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_precedence_with_integer_values(self, num_agents: int) -> None:
        """Property: Precedence works correctly for integer configuration values.
        
        Property 4: Configuration Precedence
        Validates: Requirements 2.9
        """
        import os
        
        default_value = ToxpConfig.get_defaults().num_agents
        assume(num_agents != default_value)
        
        # Ensure no env var interference
        old_env = os.environ.get("TOXP_NUM_AGENTS")
        try:
            os.environ.pop("TOXP_NUM_AGENTS", None)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config_dir = Path(tmpdir)
                manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
                
                # Set file value
                manager.set("num-agents", str(num_agents))
                
                # Load config
                config = manager.load()
                assert config.num_agents == num_agents, "File value should be used"
                
                # Apply CLI override with different value
                cli_value = 2 if num_agents > 2 else 32
                config = manager.apply_overrides(config, {"num-agents": cli_value})
                assert config.num_agents == cli_value, "CLI should override file"
                
        finally:
            if old_env is not None:
                os.environ["TOXP_NUM_AGENTS"] = old_env



class TestConfigReset:
    """Property tests for configuration reset.
    
    Property 3: Configuration Reset Restores Defaults
    Validates: Requirements 2.6
    
    For any modified configuration state, executing `config reset` SHALL restore
    all values to their documented defaults.
    """

    @given(
        provider=valid_providers,
        aws_profile=valid_aws_profiles,
        region=valid_regions,
        num_agents=valid_num_agents,
        temperature=valid_temperatures,
        log_enabled=valid_log_enabled,
        log_retention_days=valid_log_retention_days,
    )
    @settings(max_examples=100)
    def test_reset_restores_all_defaults(
        self,
        provider: str,
        aws_profile: str,
        region: str,
        num_agents: int,
        temperature: float,
        log_enabled: bool,
        log_retention_days: int,
    ) -> None:
        """Property: After reset, all config values match documented defaults.
        
        Property 3: Configuration Reset Restores Defaults
        Validates: Requirements 2.6
        """
        import os
        
        # Clear any env vars that might interfere
        env_vars_to_clear = [
            "TOXP_AWS_PROFILE", "TOXP_PROVIDER", "TOXP_REGION", 
            "TOXP_NUM_AGENTS", "TOXP_TEMPERATURE", "TOXP_LOG_ENABLED",
            "TOXP_LOG_RETENTION_DAYS"
        ]
        old_env = {k: os.environ.get(k) for k in env_vars_to_clear}
        
        try:
            for k in env_vars_to_clear:
                os.environ.pop(k, None)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config_dir = Path(tmpdir)
                manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
                
                # Modify multiple config values
                manager.set("provider", provider)
                manager.set("aws-profile", aws_profile)
                manager.set("region", region)
                manager.set("num-agents", str(num_agents))
                manager.set("temperature", str(temperature))
                manager.set("log-enabled", str(log_enabled).lower())
                manager.set("log-retention-days", str(log_retention_days))
                
                # Verify values were set
                config = manager.load()
                assert config.provider == provider
                assert config.aws_profile == aws_profile
                
                # Reset to defaults
                manager.reset()
                
                # Load and verify all values match defaults
                config = manager.load()
                defaults = ToxpConfig.get_defaults()
                
                assert config.provider == defaults.provider
                assert config.aws_profile == defaults.aws_profile
                assert config.region == defaults.region
                assert config.model == defaults.model
                assert config.num_agents == defaults.num_agents
                assert config.temperature == defaults.temperature
                assert config.coordinator_temperature == defaults.coordinator_temperature
                assert config.max_tokens == defaults.max_tokens
                assert config.log_enabled == defaults.log_enabled
                assert config.log_retention_days == defaults.log_retention_days
                
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_reset_after_single_modification(self, num_agents: int) -> None:
        """Property: Reset restores defaults even after single value modification.
        
        Property 3: Configuration Reset Restores Defaults
        Validates: Requirements 2.6
        """
        import os
        
        old_env = os.environ.get("TOXP_NUM_AGENTS")
        try:
            os.environ.pop("TOXP_NUM_AGENTS", None)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config_dir = Path(tmpdir)
                manager = ConfigManager(config_dir=config_dir, config_file=config_dir / "config.json")
                
                defaults = ToxpConfig.get_defaults()
                
                # Modify single value
                manager.set("num-agents", str(num_agents))
                
                # Reset
                manager.reset()
                
                # Verify default is restored
                config = manager.load()
                assert config.num_agents == defaults.num_agents
                
        finally:
            if old_env is not None:
                os.environ["TOXP_NUM_AGENTS"] = old_env

    def test_reset_creates_file_if_missing(self) -> None:
        """Property: Reset creates config file with defaults if it doesn't exist.
        
        Property 3: Configuration Reset Restores Defaults
        Validates: Requirements 2.6
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "config.json"
            manager = ConfigManager(config_dir=config_dir, config_file=config_file)
            
            # File shouldn't exist yet
            assert not config_file.exists()
            
            # Reset should create file with defaults
            manager.reset()
            
            # File should now exist
            assert config_file.exists()
            
            # And contain defaults
            import os
            old_env = {k: os.environ.get(k) for k in ["TOXP_AWS_PROFILE"]}
            try:
                os.environ.pop("TOXP_AWS_PROFILE", None)
                config = manager.load()
                defaults = ToxpConfig.get_defaults()
                assert config.to_dict() == defaults.to_dict()
            finally:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
