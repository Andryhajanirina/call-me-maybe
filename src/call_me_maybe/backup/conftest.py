"""Shared fixtures for token constraint tests."""

from pathlib import Path

import pytest
from llm_sdk import Small_LLM_Model

from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints


@pytest.fixture(scope="session")
def model() -> Small_LLM_Model:
    """Create one language model for the test session."""
    return Small_LLM_Model()


@pytest.fixture(scope="session")
def constraints(
    model: Small_LLM_Model,
) -> TokenConstraints:
    """Create token constraints for the test session."""
    project_root = Path(__file__).resolve().parents[2]
    schema_path = project_root / "data" / "input" / "functions_definition.json"
    schema = FunctionSchema(schema_path)
    return TokenConstraints(model, schema)
