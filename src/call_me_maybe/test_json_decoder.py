from call_me_maybe.json_decoder import JSONDecoder
# from call_me_maybe.json_state import JSONState


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


def main() -> None:
    print("=== Test sans espaces ===")
    test_nested_object()

    print("\n=== Test avec espaces ===")
    test_json_with_spaces()

    print("\n=== Test espaces ===")
    test_whitespace()

    print("\n=== Test nombre invalide avec espace ===")
    test_invalid_number_with_space()


if __name__ == "__main__":
    main()
