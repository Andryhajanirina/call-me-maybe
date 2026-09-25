"""Unit tests for JSONDecoder state machine and clone behavior."""

import pytest
from .json_decoder import JSONDecoder
from .json_state import JSONState


def test_decoder_initial_state() -> None:
    """Test that decoder starts in START state."""
    decoder = JSONDecoder()
    assert decoder.state == JSONState.START


def test_valid_json_flow() -> None:
    """Test character consumption for a minimal JSON object."""
    decoder = JSONDecoder()

    # Consuming '{'
    assert decoder.consume_char("{") is True
    assert decoder.state == JSONState.EXPECT_KEY_OR_END

    # Consuming '"'
    assert decoder.consume_char('"') is True
    assert decoder.state == JSONState.KEY_CONTENT

    # Consuming key characters 'name'
    for char in "name":
        assert decoder.consume_char(char) is True

    # Closing key quote '"'
    assert decoder.consume_char('"') is True
    assert decoder.state == JSONState.EXPECT_COLON

    # Colon ':'
    assert decoder.consume_char(":") is True
    assert decoder.state == JSONState.EXPECT_VALUE_START


def test_invalid_character_rejection() -> None:
    """Test that invalid characters return False."""
    decoder = JSONDecoder()
    assert decoder.consume_char("a") is False
    assert decoder.state == JSONState.START


def test_decoder_clone() -> None:
    """Test that cloning creates an independent copy with identical state."""
    decoder = JSONDecoder()
    decoder.consume_char("{")
    decoder.consume_char('"')

    cloned = decoder.clone()
    assert cloned.state == decoder.state
    assert cloned.stack == decoder.stack

    # Consuming on original does not affect clone
    decoder.consume_char("a")
    assert cloned.state == JSONState.KEY_CONTENT
