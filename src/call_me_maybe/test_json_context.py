from call_me_maybe.json_context import JSONContext


def test_value_context() -> None:
    context = JSONContext()

    context.start_value()
    for char in "a[eiou]":
        context.add_value_character(char)

    assert context.finish_value() == "a[eiou]"


def test_start_value_resets_previous_value() -> None:
    context = JSONContext()

    context.start_value()
    for char in "old":
        context.add_value_character(char)
    context.start_value()

    assert context.finish_value() == ""


def main() -> None:
    # context = JSONContext()
    # context.start_key()

    # for char in "name":
    #     context.add_key_character(char)

    # assert context.finish_key() == "name"

    # print("Key:", context.finish_key())

    # print("Is name:", context.is_key("name"))
    # print("Is parameters:", context.is_key("parameters"))
    # print(
    #     "Is function name:",
    #     context.is_function_name_key(),
    # )
    test_value_context()
    test_start_value_resets_previous_value()


if __name__ == "__main__":
    main()
