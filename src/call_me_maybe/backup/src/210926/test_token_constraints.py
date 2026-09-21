from llm_sdk import Small_LLM_Model

from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints
from call_me_maybe.json_decoder import JSONDecoder


def test_number_value_start_tokens() -> None:
    model = Small_LLM_Model()
    schema = FunctionSchema(
        "data/input/functions_definition.json"
    )
    constraints = TokenConstraints(model, schema)

    constraints.current_function = "fn_get_square_root"
    constraints.current_parameter = "a"

    allowed = constraints.get_allowed_value_start_tokens()

    print("\nAllowed number start tokens:", allowed)

    assert allowed == {
        12,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    }


def main() -> None:
    test_number_value_start_tokens()


def main2() -> None:
    model = Small_LLM_Model()
    schema = FunctionSchema(
        "data/input/functions_definition.json"
    )
    constraints = TokenConstraints(model, schema)
    print("\nTest process_token:")

    # generated_tokens = [8822, 1889, 3744]

    # constraints.process_token(
    #     3744,
    #     generated_tokens,
    # )

    # print("Fonction actuelle :", constraints.current_function)
    # =============================
    generated_tokens = []
    token_ids = [
        4913,
        8822,
        1889,
        3744,
        22317,
        606,
        788,
    ]

    # for token_id in [8822, 1889, 3744]:
    for token_id in token_ids:
        generated_tokens.append(token_id)

        constraints.process_token(
            token_id,
            generated_tokens,
        )

        print(
            "Tokens :",
            generated_tokens,
            "| Fonction :",
            constraints.current_function,
            "| Paramètre :",
            constraints.current_parameter,
            "| Type :",
            constraints.get_current_parameter_type(),
            "Allowed value start tokens:",
            constraints.get_allowed_value_start_tokens(),
        )
    # ============================
    # *****************************

    # texts = [
    #     "{",
    #     '"',
    #     "fn_greet",
    #     '"',
    #     ":",
    #     '{"fn_greet"',
    #     '{"fn_greet":',
    # ]

    # for text in texts:
    #     token_ids = model.encode(text).squeeze(0).tolist()

    #     print()
    #     print("Texte :", repr(text))
    #     print("Tokens:", token_ids)

    #     for token_id in token_ids:
    #         decoded = model.decode([token_id])
    #         print(" ", token_id, "->", repr(decoded))

    # *****************************

    # print("Fonction actuelle :", constraints.current_function)

    # constraints.update_current_function(
    #     [8822, 1889, 3744]
    # )

    # print("Fonction actuelle :", constraints.current_function)

    # print(
    #     constraints.get_function_name_from_tokens(
    #         [8822, 1889, 3744]
    #     )
    # )

    # print(
    #     constraints.get_function_name_from_tokens(
    #         [999]
    #     )
    # )

    # print("Test process_token_text:")
    # constraints.process_token_text('{"age":25}')

    # print("\nTest JSONDecoder avec des tokens:")
    # constraints.decoder = JSONDecoder()

    # test_token_ids = [90, 314, 341, 515, 0, 92]

    # for token_id in test_token_ids:
    #     token_text = model.decode([token_id])
    #     valid = constraints.decoder.consume_token(token_text)
    #     print(
    #         token_id,
    #         repr(token_text),
    #         "->",
    #         valid,
    #         constraints.decoder.state,
    #     )


if __name__ == "__main__":
    test_number_value_start_tokens()
    # main()
