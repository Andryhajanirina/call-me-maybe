"""Unit tests for JSONContext key tracking and cloning."""

from .json_context import JSONContext


def test_key_building() -> None:
    """Test starting, appending, and finalizing a key."""
    context = JSONContext()
    context.start_key()

    for char in "fn_add_numbers":
        context.add_key_character(char)

    completed_key = context.finish_key()
    assert completed_key == "fn_add_numbers"
    assert context.current_key == "fn_add_numbers"


def test_context_clone() -> None:
    """Test cloning JSONContext instance."""
    context = JSONContext()
    context.start_key()
    context.add_key_character("a")
    context.current_function_name = "fn_add"

    cloned = context.clone()
    assert cloned.current_key == context.current_key
    assert cloned.current_function_name == "fn_add"

    # Modify original
    context.add_key_character("b")
    assert cloned.building_key == ["a"]