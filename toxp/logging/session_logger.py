"""Session logger for saving TOXP sessions in markdown format."""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from toxp.models import Result


# Default logs directory
LOGS_DIR = Path.home() / ".toxp" / "logs" / "sessions"

# Model pricing estimates (per 1M tokens)
# Claude Sonnet: ~$3/1M input, ~$15/1M output
# Simplified estimate assuming 50/50 split = ~$9/1M tokens average
MODEL_PRICING = {
    "default": 0.000009,  # $9/1M tokens
    "claude-sonnet": 0.000009,
    "claude-haiku": 0.000001,  # $1/1M tokens
    "claude-opus": 0.000045,  # $45/1M tokens
}


class SessionLogger:
    """Logs sessions in markdown format with auto-cleanup."""

    def __init__(
        self,
        enabled: bool = True,
        retention_days: int = 30,
        logs_dir: Optional[Path] = None,
    ):
        """
        Initialize the session logger.

        Args:
            enabled: Whether logging is enabled
            retention_days: Number of days to retain logs
            logs_dir: Custom logs directory (defaults to ~/.toxp/logs/sessions/)
        """
        self.enabled = enabled
        self.retention_days = retention_days
        self.logs_dir = logs_dir or LOGS_DIR

        if self.enabled:
            self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log_session(self, result: Result) -> Optional[Path]:
        """
        Save session log and return file path.

        Args:
            result: The complete result from query processing

        Returns:
            Path to the created log file, or None if logging is disabled
        """
        if not self.enabled:
            return None

        self._cleanup_old_logs()

        # Generate filename: YYYY-MM-DD_HHMMSS_{query_id}.md
        timestamp = result.query.timestamp
        filename = f"{timestamp:%Y-%m-%d_%H%M%S}_{result.query.query_id}.md"
        filepath = self.logs_dir / filename

        content = self._format_markdown(result)
        filepath.write_text(content)

        return filepath

    def _format_markdown(self, result: Result) -> str:
        """
        Format result as markdown log.

        Args:
            result: The complete result from query processing

        Returns:
            Formatted markdown string
        """
        successful = [r for r in result.agent_responses if r.success]
        total_tokens = sum(r.token_count for r in result.agent_responses)
        model_id = result.metadata.get("model_id", "unknown")
        cost_estimate = self._estimate_cost(total_tokens, model_id)
        total_duration = result.metadata.get("total_duration_seconds", 0)

        # Build agent summary table
        agent_table = self._format_agent_table(result.agent_responses)

        return f"""# TOXP Session Log

**Query ID:** {result.query.query_id}
**Timestamp:** {result.query.timestamp.isoformat()}
**Model:** {model_id}
**Agents:** {len(successful)}/{len(result.agent_responses)} successful

## Query

{result.query.text}

## Final Answer

**Confidence:** {result.coordinator_response.confidence}

{result.coordinator_response.final_answer}

## Coordinator Synthesis

{result.coordinator_response.synthesis}

## Agent Summary

| Agent | Status | Tokens | Duration |
|-------|--------|--------|----------|
{agent_table}

## Metadata

- **Total Tokens:** {total_tokens:,}
- **Estimated Cost:** ${cost_estimate:.4f}
- **Total Duration:** {total_duration:.2f}s
- **Coordinator Duration:** {result.coordinator_response.duration_seconds:.2f}s
"""

    def _format_agent_table(self, agent_responses) -> str:
        """
        Format agent responses as markdown table rows.

        Args:
            agent_responses: List of AgentResponse objects

        Returns:
            Formatted table rows string
        """
        rows = []
        for response in agent_responses:
            status = "✓" if response.success else "✗"
            tokens = f"{response.token_count:,}" if response.token_count else "-"
            duration = (
                f"{response.duration_seconds:.2f}s"
                if response.duration_seconds
                else "-"
            )
            rows.append(f"| {response.agent_id} | {status} | {tokens} | {duration} |")
        return "\n".join(rows)

    def _cleanup_old_logs(self) -> int:
        """
        Delete logs older than retention_days.

        Returns:
            Number of files deleted
        """
        if not self.logs_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0

        for log_file in self.logs_dir.glob("*.md"):
            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff:
                    log_file.unlink()
                    deleted_count += 1
            except (OSError, ValueError):
                # Skip files we can't access or parse
                continue

        return deleted_count

    def _estimate_cost(self, tokens: int, model_id: str) -> float:
        """
        Estimate cost based on model pricing.

        Args:
            tokens: Total number of tokens used
            model_id: The model identifier

        Returns:
            Estimated cost in USD
        """
        # Determine pricing tier based on model ID
        price_per_token = MODEL_PRICING["default"]

        model_lower = model_id.lower()
        if "haiku" in model_lower:
            price_per_token = MODEL_PRICING["claude-haiku"]
        elif "opus" in model_lower:
            price_per_token = MODEL_PRICING["claude-opus"]
        elif "sonnet" in model_lower:
            price_per_token = MODEL_PRICING["claude-sonnet"]

        return tokens * price_per_token

    def get_log_path(self) -> Path:
        """Return the logs directory path."""
        return self.logs_dir

    def list_logs(self, limit: int = 10) -> list[Path]:
        """
        List recent log files.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log file paths, most recent first
        """
        if not self.logs_dir.exists():
            return []

        logs = sorted(self.logs_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        return logs[:limit]
