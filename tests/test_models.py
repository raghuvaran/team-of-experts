"""Property-based tests for TOXP data models.

Feature: toxp-cli
"""

import pytest
from hypothesis import given, strategies as st, settings

from toxp.models import CoordinatorResponse
from toxp.models.response import VALID_CONFIDENCE_LEVELS


class TestCoordinatorConfidenceLevel:
    """Property tests for coordinator confidence level validation.
    
    Property 11: Coordinator Confidence Level
    Validates: Requirements 7.4
    
    For any successful coordinator synthesis, the confidence field SHALL be
    one of: "Low", "Medium", or "High".
    """

    @given(
        synthesis=st.text(min_size=1),
        confidence=st.sampled_from(VALID_CONFIDENCE_LEVELS),
        final_answer=st.text(min_size=1),
        consensus_summary=st.text(),
        debates_summary=st.text(),
        duration_seconds=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_valid_confidence_levels_accepted(
        self,
        synthesis: str,
        confidence: str,
        final_answer: str,
        consensus_summary: str,
        debates_summary: str,
        duration_seconds: float,
    ) -> None:
        """Property: For any valid confidence level, CoordinatorResponse creation succeeds
        and the confidence field is preserved exactly.
        
        Property 11: Coordinator Confidence Level
        Validates: Requirements 7.4
        """
        response = CoordinatorResponse(
            synthesis=synthesis,
            confidence=confidence,
            final_answer=final_answer,
            consensus_summary=consensus_summary,
            debates_summary=debates_summary,
            duration_seconds=duration_seconds,
        )
        
        # The confidence level must be exactly one of the valid values
        assert response.confidence in VALID_CONFIDENCE_LEVELS
        assert response.confidence == confidence

    @given(
        synthesis=st.text(min_size=1),
        invalid_confidence=st.text().filter(lambda x: x not in VALID_CONFIDENCE_LEVELS),
        final_answer=st.text(min_size=1),
    )
    @settings(max_examples=100)
    def test_invalid_confidence_levels_rejected(
        self,
        synthesis: str,
        invalid_confidence: str,
        final_answer: str,
    ) -> None:
        """Property: For any string that is not a valid confidence level,
        CoordinatorResponse creation raises a ValidationError.
        
        Property 11: Coordinator Confidence Level
        Validates: Requirements 7.4
        """
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            CoordinatorResponse(
                synthesis=synthesis,
                confidence=invalid_confidence,
                final_answer=final_answer,
            )

    def test_all_valid_confidence_levels_documented(self) -> None:
        """Verify that the valid confidence levels match the requirements.
        
        Property 11: Coordinator Confidence Level
        Validates: Requirements 7.4
        """
        expected_levels = {"Low", "Medium", "High"}
        assert set(VALID_CONFIDENCE_LEVELS) == expected_levels
