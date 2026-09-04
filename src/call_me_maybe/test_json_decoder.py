from call_me_maybe.json_decoder import JSONDecoder
# from call_me_maybe.json_state import JSONState


def main() -> None:
    decoder = JSONDecoder()

    # tests = [
    #     "{",
    #     '"',
    #     "name",
    #     '"',
    #     ":",
    #     '"',
    #     "test",
    #     '"',
    #     "}",
    # ]

    # for token in tests:
    #     valid = decoder.consume_token(token)
    #     print(
    #         f"Token {token!r:10} "
    #         f"valid={valid:<5} "
    #         f"state={decoder.state}"
    #     )

    # text = '{"name":"fn_add_numbers"}'

    # for char in text:
    #     result = decoder.consume_char(char)
    #     print(
    #         repr(char),
    #         result,
    #         decoder.state,
    #     )

    # ===============TEST DE CONSUM TOKEN================

    print(decoder.consume_token("{"))
    print(decoder.state)

    print(decoder.consume_token('"'))
    print(decoder.state)

    print(decoder.consume_token("name"))
    print(decoder.state)

    print(decoder.consume_token('"'))
    print(decoder.state)

    print(decoder.consume_token(":"))
    print(decoder.state)

    print("space:", decoder.consume_char(" "))
    print("tab:", decoder.consume_char("\t"))
    print("newline:", decoder.consume_char("\n"))


if __name__ == "__main__":
    main()
