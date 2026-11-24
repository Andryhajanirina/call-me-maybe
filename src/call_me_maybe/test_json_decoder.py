from call_me_maybe.json_decoder import JSONDecoder
# from call_me_maybe.json_state import JSONState


def test_nested_object() -> None:
    decoder = JSONDecoder()
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
        result = decoder.consume_text(token)
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
        result = decoder.consume_text(token)
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
        assert decoder.consume_text(token) is True

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


def test_escaped_quotes() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"name"',
        ":",
        '"John \\"The Boss\\" Smith"',
        "}",
    ]

    for token in tokens:
        assert decoder.consume_text(token) is True


def test_escaped_backslashes() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"path"',
        ":",
        '"C:\\\\Users\\\\Andry\\\\Documents"',
        "}",
    ]

    for token in tokens:
        assert decoder.consume_text(token) is True


def test_unicode_escape() -> None:
    decoder = JSONDecoder()

    tokens = [
        "{",
        '"name"',
        ":",
        '"Caf\\u00e9"',
        "}",
    ]

    for token in tokens:
        assert decoder.consume_text(token) is True


def test_invalid_escape_character() -> None:
    decoder = JSONDecoder()

    assert decoder.consume_text('{"name":"John') is True
    assert decoder.consume_text('\\q') is False


def test_invalid_unicode_escape() -> None:
    decoder = JSONDecoder()

    assert decoder.consume_text('{"name":"Caf') is True
    assert decoder.consume_text('\\u12G4') is False


def test_incomplete_unicode_escape() -> None:
    decoder = JSONDecoder()

    assert decoder.consume_text('{"name":"Caf') is True
    assert decoder.consume_text('\\u12') is True
    assert decoder.consume_text('"}') is False


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


if __name__ == "__main__":
    main()
