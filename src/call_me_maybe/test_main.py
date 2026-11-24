from typing import Any, cast
from llm_sdk import Small_LLM_Model
from call_me_maybe.main import generate_string_parameter
from call_me_maybe.json_decoder import JSONDecoder
from call_me_maybe.json_state import JSONState


class FakeModel:
    """Simulate model logits for a predefined token sequence."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.position = 0

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        vocab_size = 151643
        logits = [-100.0] * vocab_size

        if self.position >= len(self.token_ids):
            raise ValueError("Unexpected model inference")

        token_id = self.token_ids[self.position]
        logits[token_id] = 100.0
        self.position += 1

        return logits


def test_tokenizer_merges_string_closing_quote_and_comma(
    constraints: Any,
) -> None:
    token_ids = constraints.tokenizer.encode('",')

    token_texts = [
        constraints.get_token_text(token_id)
        for token_id in token_ids
    ]

    assert token_texts == ['",']

    decoder = JSONDecoder()
    decoder.consume_text('{"parameters":{"name":"Andry"')

    assert decoder.state == JSONState.EXPECT_COMMA

    assert not decoder.consume_text('",')


def test_generate_string_parameter(
    constraints: Any,
) -> None:
    token_ids = (
        constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode("Andry")
        + constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(",")
    )

    fake_model = FakeModel(token_ids)

    result = generate_string_parameter(
        cast(Small_LLM_Model, fake_model),
        "Say hello to Andry",
        "fn_greet",
        "name",
        constraints,
    )

    assert result == "Andry"
    assert fake_model.position == len(token_ids)


def test_generate_string_parameter_multiple_words(
    constraints: Any,
) -> None:
    expected = "John Smith"
    token_ids = (
        constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(expected)
        + constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(",")
    )

    fake_model = FakeModel(token_ids)

    result = generate_string_parameter(
        cast(Small_LLM_Model, fake_model),
        "Greet John Smith",
        "fn_greet",
        "name",
        constraints,
    )

    assert result == expected
    assert fake_model.position == len(token_ids)


def test_generate_string_parameter_with_comma(
    constraints: Any,
) -> None:
    expected = "hello, world!"
    token_ids = (
        constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(expected)
        + constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(",")
    )

    fake_model = FakeModel(token_ids)

    result = generate_string_parameter(
        cast(Small_LLM_Model, fake_model),
        "Reverse the string hello, world!",
        "fn_reverse_string",
        "s",
        constraints,
    )

    assert result == expected
    assert fake_model.position == len(token_ids)


def test_generate_string_parameter_with_escaped_quote(
    constraints: Any,
) -> None:
    expected = 'John "The Boss" Smith'
    token_ids = (
        constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode('John \\"The Boss\\" Smith')
        + constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(",")
    )

    fake_model = FakeModel(token_ids)

    result = generate_string_parameter(
        cast(Small_LLM_Model, fake_model),
        'Greet John "The Boss" Smith',
        "fn_greet",
        "name",
        constraints,
    )

    assert result == expected
    assert fake_model.position == len(token_ids)


def test_generate_string_parameter_with_backslash(
    constraints: Any,
) -> None:
    expected = r"C:\Users\Andry\Documents"
    token_ids = (
        constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(r"C:\\Users\\Andry\\Documents")
        + constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(",")
    )

    fake_model = FakeModel(token_ids)

    result = generate_string_parameter(
        cast(Small_LLM_Model, fake_model),
        "Return the Windows path C:\\Users\\Andry\\Documents",
        "fn_reverse_string",
        "s",
        constraints,
    )

    assert result == expected
    assert fake_model.position == len(token_ids)


def test_generate_string_parameter_with_unicode_escape(
    constraints: Any,
) -> None:
    expected = "Café"

    token_ids = (
        constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode("Caf\\u00e9")
        + constraints.tokenizer.encode('"')
        + constraints.tokenizer.encode(",")
    )

    fake_model = FakeModel(token_ids)

    result = generate_string_parameter(
        cast(Small_LLM_Model, fake_model),
        "Return the word Café",
        "fn_reverse_string",
        "s",
        constraints,
    )

    assert result == expected
    assert fake_model.position == len(token_ids)
