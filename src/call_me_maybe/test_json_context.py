from call_me_maybe.json_context import JSONContext


def main() -> None:
    context = JSONContext()
    context.start_key()

    for char in "name":
        context.add_key_character(char)

    assert context.finish_key() == "name"

    print("Key:", context.finish_key())

    print("Is name:", context.is_key("name"))
    print("Is parameters:", context.is_key("parameters"))
    print(
        "Is function name:",
        context.is_function_name_key(),
    )


if __name__ == "__main__":
    main()
