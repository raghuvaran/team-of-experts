"""Property-based tests for TOXP session logging.

Feature: toxp-cli
"""

import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings, assume

from toxp.logging import SessionLogger
from toxp.models import Query, AgentResponse, CoordinatorResponse, Result


# Strategies for generating valid test data
# Filter out carriage returns and surrogate characters as they cause encoding issues
valid_query_text = st.text(
    min_size=1,
    max_size=500,
    alphabet=st.characters(
        blacklist_characters="\r",
        blacklist_categories=("Cs",),  # Exclude surrogate characters
    ),
).filter(lambda x: x.strip())
valid_query_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=8,
    max_size=8,
)
valid_confidence_levels = st.sampled_from(["Low", "Medium", "High"])
valid_agent_ids = st.integers(min_value=0, max_value=31)
valid_token_counts = st.integers(min_value=0, max_value=100000)
valid_durations = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)
valid_retention_days = st.integers(min_value=1, max_value=365)
valid_model_ids = st.sampled_from([
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-3-5-20241022-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "unknown-model",
])


def create_mock_agent_response(
    agent_id: int = 0,
    success: bool = True,
    token_count: int = 1000,
    duration: float = 2.5,
) -> AgentResponse:
    """Create a mock agent response for testing."""
    return AgentResponse(
        agent_id=agent_id,
        success=success,
        chain_of_thought="Test reasoning" if success else "",
        final_answer="Test answer" if success else "",
        error=None if success else "Test error",
        duration_seconds=duration,
        token_count=token_count,
    )


def create_mock_coordinator_response(
    confidence: str = "High",
    duration: float = 3.0,
) -> CoordinatorResponse:
    """Create a mock coordinator response for testing."""
    return CoordinatorResponse(
        synthesis="Test synthesis content",
        confidence=confidence,
        consensus_summary="Test consensus",
        debates_summary="Test debates",
        final_answer="Test final answer",
        duration_seconds=duration,
    )


def create_mock_result(
    query_text: str = "Test query",
    query_id: str = "abc12345",
    num_agents: int = 4,
    success_rate: float = 1.0,
    confidence: str = "High",
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
) -> Result:
    """Create a mock result for testing."""
    num_success = int(num_agents * success_rate)
    agent_responses = [
        create_mock_agent_response(agent_id=i, success=(i < num_success))
        for i in range(num_agents)
    ]
    
    return Result(
        query=Query(text=query_text, query_id=query_id),
        agent_responses=agent_responses,
        coordinator_response=create_mock_coordinator_response(confidence=confidence),
        metadata={
            "model_id": model_id,
            "total_duration_seconds": 10.5,
        },
    )


class TestLogFileCreation:
    """Property tests for log file creation.
    
    Property 12: Log File Creation
    Validates: Requirements 8.1, 8.2
    
    For any logged session with logging enabled, the log file SHALL:
    - Be created in `~/.toxp/logs/sessions/` (or custom logs_dir)
    - Have filename matching pattern `YYYY-MM-DD_HHMMSS_{query_id}.md`
    """

    @given(
        query_text=valid_query_text,
        query_id=valid_query_ids,
        confidence=valid_confidence_levels,
    )
    @settings(max_examples=100)
    def test_log_file_created_in_correct_directory(
        self, query_text: str, query_id: str, confidence: str
    ) -> None:
        """Property: Log files are created in the configured logs directory.
        
        Property 12: Log File Creation
        Validates: Requirements 8.1, 8.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(
                query_text=query_text,
                query_id=query_id,
                confidence=confidence,
            )
            
            filepath = logger.log_session(result)
            
            assert filepath is not None
            assert filepath.exists()
            assert filepath.parent == logs_dir

    @given(
        query_id=valid_query_ids,
    )
    @settings(max_examples=100)
    def test_log_filename_matches_pattern(self, query_id: str) -> None:
        """Property: Log filename matches YYYY-MM-DD_HHMMSS_{query_id}.md pattern.
        
        Property 12: Log File Creation
        Validates: Requirements 8.1, 8.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(query_id=query_id)
            
            filepath = logger.log_session(result)
            
            assert filepath is not None
            
            # Verify filename pattern: YYYY-MM-DD_HHMMSS_{query_id}.md
            filename = filepath.name
            pattern = r"^\d{4}-\d{2}-\d{2}_\d{6}_" + re.escape(query_id) + r"\.md$"
            assert re.match(pattern, filename), f"Filename '{filename}' doesn't match expected pattern"

    @given(query_text=valid_query_text)
    @settings(max_examples=100)
    def test_logging_disabled_returns_none(self, query_text: str) -> None:
        """Property: When logging is disabled, log_session returns None.
        
        Property 12: Log File Creation
        Validates: Requirements 8.6
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=False, logs_dir=logs_dir)
            
            result = create_mock_result(query_text=query_text)
            
            filepath = logger.log_session(result)
            
            assert filepath is None
            # Directory should not be created when disabled
            assert not logs_dir.exists()

    @given(query_text=valid_query_text)
    @settings(max_examples=100)
    def test_log_file_is_markdown(self, query_text: str) -> None:
        """Property: Log files have .md extension and contain valid markdown.
        
        Property 12: Log File Creation
        Validates: Requirements 8.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(query_text=query_text)
            
            filepath = logger.log_session(result)
            
            assert filepath is not None
            assert filepath.suffix == ".md"
            
            # Verify it starts with markdown header
            content = filepath.read_text()
            assert content.startswith("# TOXP Session Log")



class TestLogContentCompleteness:
    """Property tests for log content completeness.
    
    Property 13: Log Content Completeness
    Validates: Requirements 8.3, 8.4
    
    For any session log file, the markdown content SHALL include:
    - Query text
    - Final synthesized answer
    - Confidence level
    - Agent summary table
    - Token counts
    - Cost estimate
    - Model version
    - Timestamp
    """

    @given(
        query_text=valid_query_text,
        confidence=valid_confidence_levels,
        model_id=valid_model_ids,
    )
    @settings(max_examples=100)
    def test_log_contains_query_text(
        self, query_text: str, confidence: str, model_id: str
    ) -> None:
        """Property: Log content includes the original query text.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(
                query_text=query_text,
                confidence=confidence,
                model_id=model_id,
            )
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            assert query_text in content

    @given(confidence=valid_confidence_levels)
    @settings(max_examples=100)
    def test_log_contains_confidence_level(self, confidence: str) -> None:
        """Property: Log content includes the confidence level.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(confidence=confidence)
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            assert f"**Confidence:** {confidence}" in content

    @given(model_id=valid_model_ids)
    @settings(max_examples=100)
    def test_log_contains_model_version(self, model_id: str) -> None:
        """Property: Log content includes the model version.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.4
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(model_id=model_id)
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            assert f"**Model:** {model_id}" in content

    @given(
        num_agents=st.integers(min_value=2, max_value=16),
        token_count=valid_token_counts,
    )
    @settings(max_examples=100)
    def test_log_contains_token_counts(
        self, num_agents: int, token_count: int
    ) -> None:
        """Property: Log content includes token counts.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.4
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            # Create result with specific token counts
            agent_responses = [
                create_mock_agent_response(agent_id=i, token_count=token_count)
                for i in range(num_agents)
            ]
            
            result = Result(
                query=Query(text="Test query"),
                agent_responses=agent_responses,
                coordinator_response=create_mock_coordinator_response(),
                metadata={"model_id": "test-model", "total_duration_seconds": 10.0},
            )
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            total_tokens = num_agents * token_count
            assert "**Total Tokens:**" in content
            assert f"{total_tokens:,}" in content

    @given(query_text=valid_query_text)
    @settings(max_examples=100)
    def test_log_contains_cost_estimate(self, query_text: str) -> None:
        """Property: Log content includes cost estimate.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.4
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(query_text=query_text)
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            assert "**Estimated Cost:** $" in content

    @given(query_text=valid_query_text)
    @settings(max_examples=100)
    def test_log_contains_timestamp(self, query_text: str) -> None:
        """Property: Log content includes timestamp.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.4
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(query_text=query_text)
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            assert "**Timestamp:**" in content
            # Verify ISO format timestamp is present
            assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", content)

    @given(num_agents=st.integers(min_value=2, max_value=16))
    @settings(max_examples=100)
    def test_log_contains_agent_summary_table(self, num_agents: int) -> None:
        """Property: Log content includes agent summary table.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(num_agents=num_agents)
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            # Verify table header
            assert "| Agent | Status | Tokens | Duration |" in content
            assert "|-------|--------|--------|----------|" in content
            
            # Verify each agent has a row
            for i in range(num_agents):
                assert f"| {i} |" in content

    @given(query_text=valid_query_text)
    @settings(max_examples=100)
    def test_log_contains_final_answer(self, query_text: str) -> None:
        """Property: Log content includes the final synthesized answer.
        
        Property 13: Log Content Completeness
        Validates: Requirements 8.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logger = SessionLogger(enabled=True, logs_dir=logs_dir)
            
            result = create_mock_result(query_text=query_text)
            
            filepath = logger.log_session(result)
            content = filepath.read_text()
            
            assert "## Final Answer" in content
            assert result.coordinator_response.final_answer in content



class TestLogRetentionCleanup:
    """Property tests for log retention cleanup.
    
    Property 14: Log Retention Cleanup
    Validates: Requirements 8.5
    
    For any log file older than the configured retention_days, the cleanup
    process SHALL delete that file.
    """

    @given(
        retention_days=valid_retention_days,
        file_age_days=st.integers(min_value=0, max_value=400),
    )
    @settings(max_examples=100)
    def test_old_logs_are_deleted(
        self, retention_days: int, file_age_days: int
    ) -> None:
        """Property: Files older than retention_days are deleted during cleanup.
        
        Property 14: Log Retention Cleanup
        Validates: Requirements 8.5
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a log file with specific age
            old_file = logs_dir / "2020-01-01_120000_testid01.md"
            old_file.write_text("# Old log content")
            
            # Set file modification time to simulate age
            old_time = datetime.now() - timedelta(days=file_age_days)
            os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
            
            # Create logger and trigger cleanup
            logger = SessionLogger(
                enabled=True,
                retention_days=retention_days,
                logs_dir=logs_dir,
            )
            
            # Cleanup is triggered during log_session
            result = create_mock_result()
            logger.log_session(result)
            
            # Check if old file was deleted based on age vs retention
            should_be_deleted = file_age_days >= retention_days
            file_exists = old_file.exists()
            
            if should_be_deleted:
                assert not file_exists, f"File aged {file_age_days} days should be deleted (retention={retention_days})"
            else:
                assert file_exists, f"File aged {file_age_days} days should be kept (retention={retention_days})"

    @given(retention_days=valid_retention_days)
    @settings(max_examples=100)
    def test_recent_logs_are_preserved(self, retention_days: int) -> None:
        """Property: Files newer than retention_days are preserved.
        
        Property 14: Log Retention Cleanup
        Validates: Requirements 8.5
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a recent log file (within retention period)
            recent_file = logs_dir / "2024-12-30_120000_recent01.md"
            recent_file.write_text("# Recent log content")
            
            # Set file modification time to be recent (within retention)
            recent_time = datetime.now() - timedelta(days=retention_days // 2)
            os.utime(recent_file, (recent_time.timestamp(), recent_time.timestamp()))
            
            # Create logger and trigger cleanup
            logger = SessionLogger(
                enabled=True,
                retention_days=retention_days,
                logs_dir=logs_dir,
            )
            
            result = create_mock_result()
            logger.log_session(result)
            
            # Recent file should still exist
            assert recent_file.exists(), "Recent files should be preserved"

    @given(
        retention_days=valid_retention_days,
        num_old_files=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_multiple_old_logs_deleted(
        self, retention_days: int, num_old_files: int
    ) -> None:
        """Property: Multiple old files are all deleted during cleanup.
        
        Property 14: Log Retention Cleanup
        Validates: Requirements 8.5
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create multiple old log files
            old_files = []
            for i in range(num_old_files):
                old_file = logs_dir / f"2020-01-0{i+1}_120000_oldfile{i:02d}.md"
                old_file.write_text(f"# Old log content {i}")
                
                # Set file modification time to be older than retention
                old_time = datetime.now() - timedelta(days=retention_days + 10 + i)
                os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
                old_files.append(old_file)
            
            # Create logger and trigger cleanup
            logger = SessionLogger(
                enabled=True,
                retention_days=retention_days,
                logs_dir=logs_dir,
            )
            
            result = create_mock_result()
            logger.log_session(result)
            
            # All old files should be deleted
            for old_file in old_files:
                assert not old_file.exists(), f"Old file {old_file.name} should be deleted"

    @given(retention_days=valid_retention_days)
    @settings(max_examples=100)
    def test_cleanup_returns_deleted_count(self, retention_days: int) -> None:
        """Property: Cleanup returns the count of deleted files.
        
        Property 14: Log Retention Cleanup
        Validates: Requirements 8.5
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "sessions"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create some old files
            num_old = 3
            for i in range(num_old):
                old_file = logs_dir / f"2020-01-0{i+1}_120000_oldfile{i:02d}.md"
                old_file.write_text(f"# Old log content {i}")
                old_time = datetime.now() - timedelta(days=retention_days + 10)
                os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
            
            # Create some recent files
            num_recent = 2
            for i in range(num_recent):
                recent_file = logs_dir / f"2024-12-3{i}_120000_recent{i:02d}.md"
                recent_file.write_text(f"# Recent log content {i}")
            
            logger = SessionLogger(
                enabled=True,
                retention_days=retention_days,
                logs_dir=logs_dir,
            )
            
            deleted_count = logger._cleanup_old_logs()
            
            assert deleted_count == num_old
