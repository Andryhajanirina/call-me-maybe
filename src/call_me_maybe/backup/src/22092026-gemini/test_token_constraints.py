"""Unit tests for TokenConstraints logic and masking."""

from unittest.mock import MagicMock
import pytest

from .token_constraints import TokenConstraints
from .function_schema import FunctionSchema


@pytest.fixture
def dummy_schema() -> FunctionSchema:
    """Fixture providing a mock FunctionSchema."""
    schema = MagicMock(spec=FunctionSchema)
    schema.get_function_names.return_value = ["fn_add_numbers", "fn_greet"]
    return schema


@pytest.fixture
def dummy_vocab() -> dict[int, str]:
    """Fixture providing a mock token vocabulary."""
    return {
        0: "{",
        1: '"',
        2: "name",
        3: "}",
        4: "invalid_char",
    }


def test_apply_token_constraint(dummy_schema: FunctionSchema, dummy_vocab: dict[int, str]) -> None:
    """Test that allowed token IDs preserve logits and others become -inf."""
    mock_model = MagicMock()
    constraints = TokenConstraints(mock_model, dummy_schema, dummy_vocab)

    logits = [1.0, 2.0, 3.0, 4.0, 5.0]
    allowed_ids = {0, 2}

    constrained = constraints.apply_token_constraint(logits, allowed_ids)

    assert constrained[0] == 1.0
    assert constrained[1] == float("-inf")
    assert constrained[2] == 3.0
    assert constrained[3] == float("-inf")
    assert constrained[4] == float("-inf")


def test_is_token_valid_for_decoder(dummy_schema: FunctionSchema, dummy_vocab: dict[int, str]) -> None:
    """Test simulation of candidate tokens via decoder cloning."""
    mock_model = MagicMock()
    constraints = TokenConstraints(mock_model, dummy_schema, dummy_vocab)

    # At start, '{' is valid, 'name' is invalid
    assert constraints.is_token_valid_for_decoder("{") is True
    assert constraints.is_token_valid_for_decoder("name") is False


def test_get_valid_tokens_for_current_state(
    dummy_schema: FunctionSchema, dummy_vocab: dict[int, str]
) -> None:
    """Test filtering vocabulary for valid tokens at initial state."""
    mock_model = MagicMock()
    constraints = TokenConstraints(mock_model, dummy_schema, dummy_vocab)

    valid_tokens = constraints.get_valid_tokens_for_current_state()

    # Only token ID 0 ('{') is valid at start
    assert 0 in valid_tokens
    assert 1 not in valid_tokens
    assert 4 not in valid_tokens
