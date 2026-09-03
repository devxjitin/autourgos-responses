"""
Tests for normalize_reasoning_effort (autourgos_responses.core).
"""

import pytest

from autourgos_responses.core import normalize_reasoning_effort


def test_none_input_returns_none():
    assert normalize_reasoning_effort(None) is None


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_accepts_every_value_the_real_api_supports(effort):
    """
    Regression: the whitelist used to only allow low/medium/high and rejected
    none/minimal/xhigh/max, even though the real Responses API's
    reasoning.effort field documents all seven as currently supported.
    """
    assert normalize_reasoning_effort(effort) == effort


def test_normalizes_case_and_whitespace():
    assert normalize_reasoning_effort("  MINIMAL  ") == "minimal"


def test_rejects_genuinely_invalid_value():
    with pytest.raises(ValueError):
        normalize_reasoning_effort("ultra")
