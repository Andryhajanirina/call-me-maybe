from call_me_maybe.json_decoder import JSONDecoder
from call_me_maybe.json_state import JSONState


def test_nested_object() -> None:
    decoder = JSONDecoder()
    # tokens = [
    #     "{",
    #     '"name"',
    #     ":",
    #     '"fn_add_numbers"',
    #     ",",
    #     '"parameters"',
    #     ":",
    #     "{",
    #     '"a"',
    #     ":",
    #     '"number"',
    #     ",",
    #     '"b"',
    #     ":",
    #     '"number"',
    #     "}",
    #     "}"
    # ]
    tokens = [
        "{",
        '"a"',
        ":",
        "265",
        ",",
        '"b"',
        ":",
        "345",
        "}",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )
        print("    stack:", decoder.object_stack)


def test_json_with_spaces() -> None:
    decoder = JSONDecoder()
    tokens = [
        "{",
        '"a"',
        ":",
        " ",
        "265",
        ",",
        " ",
        '"b"',
        ":",
        " ",
        "345",
        " ",
        "}",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )
        print("    stack:", decoder.object_stack)


def test_whitespace() -> None:
    decoder = JSONDecoder()

    whitespace_chars = [" ", "\t", "\n", "\r"]

    for char in whitespace_chars:
        print(
            f"{char!r:5} -> "
            f"{decoder.is_whitespace(char)}"
            )


def test_invalid_number_with_space() -> None:
    decoder = JSONDecoder()
    tokens = [
        "{",
        '"a"',
        ":",
        "26",
        " ",
        "5",
        "}",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )
        print("    stack:", decoder.object_stack)


def test_partial_invalid_token() -> None:
    decoder = JSONDecoder()
    tokens = [
        "{",
        '"a"',
        ":",
    ]

    for token in tokens:
        assert decoder.consume_token(token) is True

    print("Before invalid token:", decoder.state, decoder.object_stack)

    result = decoder.consume_token("26 5")

    print("Result:", result)
    print("After invalid token:", decoder.state, decoder.object_stack)


def test_after_done() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"a"',
        ":",
        "265",
        "}",
    ]

    for token in tokens:
        assert decoder.consume_token(token) is True

    print("State after valid JSON:", decoder.state)

    result = decoder.consume_token(" ")
    print("Token after DONE:", result)
    print("State after attempt:", decoder.state)


def test_spaces_after_string() -> None:
    decoder = JSONDecoder()
    tokens = [
        "{",
        '"name"',
        ":",
        '"andry"',
        " ",
        "}",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )
        print("    stack:", decoder.object_stack)


def test_multi_character_token() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"a"',
        ":",
        "345",
        " }",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )
        print("    stack:", decoder.object_stack)


def test_number_followed_by_space() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"a"',
        ":",
        "345",
        " ",
        "}",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )


def test_number_followed_by_comma() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"a"',
        ":",
        "345",
        " ",
        ",",
        '"b"',
        ":",
        "10",
        "}",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )


def test_invalid_number_space_digit() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"a"',
        ":",
        "345",
        " ",
        "6",
    ]

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )


def test_function_call_object() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"fn_greet"',
        ":",
        "{",
        '"name"',
        ":",
        '"Andry"',
        "}",
        "}",
    ]
    for token in tokens:
        result = decoder.consume_token(token)
        print(
            f"{token!r:20} -> {result} -> {decoder.state}"
        )
        print("    stack:", decoder.object_stack)

    assert result is True


def test_allowed_number_characters() -> None:
    decoder = JSONDecoder()

    print(
        "START:",
        decoder.get_allowed_number_characters()
    )

    decoder.consume_char("{")
    decoder.consume_char('"')
    decoder.consume_char("a")
    decoder.consume_char('"')
    decoder.consume_char(":")

    print(
        "EXPECT_VALUE_START:",
        decoder.get_allowed_number_characters()
    )

    assert decoder.state == JSONState.EXPECT_VALUE_START


def main() -> None:

    tokens = [
        '{', '"a"', ':', '0',
        ',', '"b"', ':', '5',
        '}'
    ]

    decoder = JSONDecoder()

    for token in tokens:
        result = decoder.consume_token(token)
        print(
            repr(token),
            "->",
            result,
            "->",
            decoder.state,
            "stack:",
            decoder.object_stack,
        )

    print("\n=== Test function_call_object ===")
    test_function_call_object()
    print("\n=== Test test_allowed_number_characters ===")
    test_allowed_number_characters()


if __name__ == "__main__":
    main()
