import json

import pytest

from call_me_maybe.function_schema import FunctionSchema
from pathlib import Path
from typing import Any


@pytest.fixture
def schema(tmp_path: Path) -> FunctionSchema:
    data = [
        {
            "name": "fn_test",
            "description": "Test function",
            "parameters": {
                "number_value": {"type": "number"},
                "int_value": {"type": "int"},
                "integer_value": {"type": "integer"},
                "string_value": {"type": "string"},
                "boolean_value": {"type": "boolean"},
                "array_value": {"type": "array"},
                "object_value": {"type": "object"},
            },
            "returns": {"type": "number"},
        }
    ]

    path = tmp_path / "functions.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    return FunctionSchema(path)


def normalize_with_defaults(
        schema: FunctionSchema,
        parameters: dict[str, Any],
) -> dict[str, Any]:
    defaults = {
        "number_value": 2,
        "int_value": 2,
        "integer_value": 2,
        "string_value": "hello",
        "boolean_value": True,
        "array_value": [1, 2],
        "object_value": {"key": "value"},
    }
    defaults.update(parameters)

    return schema.normalize_parameters("fn_test", defaults)


def test_number_is_normalized_to_float(schema: FunctionSchema) -> None:
    result = normalize_with_defaults(
        schema,
        {"number_value": 2},
    )

    assert result["number_value"] == 2.0
    assert isinstance(result["number_value"], float)


@pytest.mark.parametrize("value", [2, 25.0])
def test_integer_types_are_normalized_to_int(
    schema: FunctionSchema,
    value: int | float,
) -> None:
    result = normalize_with_defaults(
        schema,
        {
            "int_value": value,
            "integer_value": value,
        },
    )

    assert isinstance(result["int_value"], int)
    assert isinstance(result["integer_value"], int)


def test_fractional_value_is_rejected_for_integer(
        schema: FunctionSchema
) -> None:
    with pytest.raises(ValueError):
        schema.normalize_parameters(
            "fn_test",
            {"int_value": 2.5},
        )


def test_boolean_is_rejected_as_number(
        schema: FunctionSchema
) -> None:
    with pytest.raises(ValueError):
        schema.normalize_parameters(
            "fn_test",
            {"number_value": True},
        )


def test_other_types_are_preserved(
        schema: FunctionSchema
) -> None:
    result = normalize_with_defaults(
        schema,
        {
            "string_value": "hello",
            "boolean_value": True,
            "array_value": [1, 2],
            "object_value": {"key": "value"},
        },
    )

    assert result["string_value"] == "hello"
    assert result["boolean_value"] is True
    assert result["array_value"] == [1, 2]
    assert result["object_value"] == {"key": "value"}


def test_unknown_function_is_rejected(
        schema: FunctionSchema
) -> None:
    with pytest.raises(ValueError):
        schema.normalize_parameters("unknown", {})


def test_unknown_parameter_is_rejected(
        schema: FunctionSchema
) -> None:
    with pytest.raises(ValueError):
        schema.normalize_parameters(
            "fn_test",
            {"unknown_value": 10},
        )


def test_missing_parameter_is_rejected(
        schema: FunctionSchema
) -> None:
    with pytest.raises(ValueError, match="Missing parameter"):
        schema.normalize_parameters(
            "fn_test",
            {"number_value": 2},
        )
