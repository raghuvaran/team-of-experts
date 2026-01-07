"""Custom exceptions for TOXP CLI.

This module defines a hierarchy of exceptions for the TOXP CLI tool,
providing user-friendly error messages and formatting.

Feature: toxp-cli
Requirements: 10.1, 10.2, 10.3, 10.4, 10.6, 10.7
"""

from typing import List, Optional


class ToxpError(Exception):
    """Base exception for all TOXP errors.
    
    Provides a consistent interface for user-friendly error formatting
    and optional suggestions for resolution.
    
    Attributes:
        message: The error message
        suggestion: Optional suggestion for resolving the error
        details: Optional additional details
    """
    
    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        details: Optional[str] = None,
    ):
        """Initialize ToxpError.
        
        Args:
            message: The main error message
            suggestion: Optional suggestion for resolving the error
            details: Optional additional technical details
        """
        self.message = message
        self.suggestion = suggestion
        self.details = details
        super().__init__(message)
    
    def format_for_user(self, verbose: bool = False) -> str:
        """Format the error for user display.
        
        Args:
            verbose: If True, include additional details
            
        Returns:
            Formatted error string suitable for CLI output
        """
        parts = [f"Error: {self.message}"]
        
        if self.suggestion:
            parts.append(f"\nSuggestion: {self.suggestion}")
        
        if verbose and self.details:
            parts.append(f"\nDetails: {self.details}")
        
        return "".join(parts)


class ConfigurationError(ToxpError):
    """Raised when there is a configuration error.
    
    This includes invalid configuration values, missing required
    configuration, or configuration file issues.
    """
    
    def __init__(
        self,
        message: str,
        key: Optional[str] = None,
        valid_values: Optional[List[str]] = None,
    ):
        """Initialize ConfigurationError.
        
        Args:
            message: The error message
            key: The configuration key that caused the error
            valid_values: Optional list of valid values for the key
        """
        self.key = key
        self.valid_values = valid_values
        
        suggestion = None
        if valid_values:
            suggestion = f"Valid values for '{key}': {', '.join(valid_values)}"
        elif key:
            suggestion = f"Check the value for configuration key '{key}'"
        
        super().__init__(message, suggestion=suggestion)


class CredentialsError(ToxpError):
    """Raised when AWS credentials are missing, invalid, or expired.
    
    Requirements: 10.1
    """
    
    def __init__(self, message: str, details: Optional[str] = None):
        """Initialize CredentialsError.
        
        Args:
            message: The error message
            details: Optional additional details about the error
        """
        # Detect expired token from details
        is_expired = details and "expired" in details.lower()
        
        if is_expired:
            suggestion = (
                "Your AWS credentials have expired. Refresh them:\n"
                "  • If using SSO: aws sso login --profile your-profile\n"
                "  • If using temporary creds: request new credentials\n"
                "  • Then retry your command"
            )
        else:
            suggestion = (
                "Configure AWS credentials:\n"
                "  1. aws configure --profile your-profile\n"
                "  2. toxp config set aws-profile your-profile\n"
                "  3. Verify: aws sts get-caller-identity --profile your-profile"
            )
        super().__init__(message, suggestion=suggestion, details=details)


class ProviderError(ToxpError):
    """Raised when there is an error with the LLM provider.
    
    This is a general error for provider-related issues that don't
    fit into more specific categories.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[str] = None,
    ):
        """Initialize ProviderError.
        
        Args:
            message: The error message
            provider: The provider that caused the error
            details: Optional additional details
        """
        self.provider = provider
        
        suggestion = None
        if provider:
            suggestion = f"Check your {provider} provider configuration"
        
        super().__init__(message, suggestion=suggestion, details=details)


class ModelNotFoundError(ToxpError):
    """Raised when the specified model is not available.
    
    Requirements: 10.2
    """
    
    def __init__(
        self,
        model_id: str,
        region: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ):
        """Initialize ModelNotFoundError.
        
        Args:
            model_id: The model ID that was not found
            region: The AWS region where the model was searched
            available_models: Optional list of available models
        """
        self.model_id = model_id
        self.region = region
        self.available_models = available_models
        
        message = f"Model not found: {model_id}"
        if region:
            message += f" in region {region}"
        
        suggestion_parts = ["Verify the model ID is correct"]
        if region:
            suggestion_parts.append(f"Check if the model is available in region: {region}")
        suggestion_parts.append("Ensure you have access to this model in your AWS account")
        
        if available_models:
            suggestion_parts.append(f"Available models: {', '.join(available_models[:5])}")
        
        suggestion = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(suggestion_parts))
        
        super().__init__(message, suggestion=suggestion)


class ThrottlingError(ToxpError):
    """Raised when rate limiting occurs despite retries.
    
    Requirements: 10.3
    """
    
    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        num_agents: Optional[int] = None,
    ):
        """Initialize ThrottlingError.
        
        Args:
            message: The error message
            retry_count: Number of retries attempted
            num_agents: Current number of agents configured
        """
        self.retry_count = retry_count
        self.num_agents = num_agents
        
        suggestion_parts = []
        if num_agents and num_agents > 4:
            suggestion_parts.append(f"Reduce num-agents (currently {num_agents}): toxp config set num-agents {max(2, num_agents // 2)}")
        suggestion_parts.append("Wait a few minutes before retrying")
        suggestion_parts.append("Check your AWS Bedrock quota limits")
        
        suggestion = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(suggestion_parts))
        
        details = f"Retries attempted: {retry_count}"
        
        super().__init__(message, suggestion=suggestion, details=details)


class InsufficientAgentsError(ToxpError):
    """Raised when too few agents succeed to produce a reliable result.
    
    Requirements: 6.5
    """
    
    def __init__(
        self,
        successful_count: int,
        total_count: int,
        min_required: int,
        agent_errors: Optional[List[str]] = None,
    ):
        """Initialize InsufficientAgentsError.
        
        Args:
            successful_count: Number of agents that succeeded
            total_count: Total number of agents spawned
            min_required: Minimum number required for synthesis
            agent_errors: List of error messages from failed agents
        """
        self.successful_count = successful_count
        self.total_count = total_count
        self.min_required = min_required
        self.agent_errors = agent_errors or []
        
        success_rate = successful_count / total_count if total_count > 0 else 0
        message = (
            f"Insufficient agents succeeded: {successful_count}/{total_count} "
            f"({success_rate:.0%}), minimum required: {min_required} (50%)"
        )
        
        suggestion = (
            "Try reducing num-agents or check your API quotas.\n"
            "  Use: toxp config set num-agents <lower_value>"
        )
        
        details = None
        if agent_errors:
            unique_errors = list(set(agent_errors[:5]))  # Dedupe and limit
            details = "Agent errors:\n" + "\n".join(f"  - {e}" for e in unique_errors)
        
        super().__init__(message, suggestion=suggestion, details=details)


class NetworkError(ToxpError):
    """Raised when there is a network connectivity issue.
    
    Requirements: 10.6
    """
    
    def __init__(self, message: str, details: Optional[str] = None):
        """Initialize NetworkError.
        
        Args:
            message: The error message
            details: Optional additional details
        """
        suggestion = (
            "Check your network connection:\n"
            "  1. Verify internet connectivity\n"
            "  2. Check if AWS services are accessible\n"
            "  3. Verify any proxy or firewall settings\n"
            "  4. Try again in a few moments"
        )
        super().__init__(message, suggestion=suggestion, details=details)


class TimeoutError(ToxpError):
    """Raised when an operation times out.
    
    Requirements: 10.7
    """
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        partial_results: bool = False,
    ):
        """Initialize TimeoutError.
        
        Args:
            message: The error message
            timeout_seconds: The timeout value that was exceeded
            partial_results: Whether partial results are available
        """
        self.timeout_seconds = timeout_seconds
        self.partial_results = partial_results
        
        suggestion_parts = ["Try again - the service may be temporarily slow"]
        if timeout_seconds:
            suggestion_parts.append(f"Current timeout: {timeout_seconds}s")
        if partial_results:
            suggestion_parts.append("Partial results may be available in the output")
        
        suggestion = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(suggestion_parts))
        
        super().__init__(message, suggestion=suggestion)
