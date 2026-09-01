from call_me_maybe.json_decoder import JSONDecoder
# from call_me_maybe.json_state import JSONState


def main() -> None:
    decoder = JSONDecoder()

    tests = [
        "{",
        '"',
        "name",
        '"',
        ":",
        '"',
        "test",
        '"',
        "}",
    ]

    for token in tests:
        valid = decoder.consume_token(token)
        print(
            f"Token {token!r:10} "
            f"valid={valid:<5} "
            f"state={decoder.state}"
        )


if __name__ == "__main__":
    main()
