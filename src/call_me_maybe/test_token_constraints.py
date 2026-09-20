from llm_sdk import Small_LLM_Model

from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints
from call_me_maybe.json_decoder import JSONDecoder


def main() -> None:
    model = Small_LLM_Model()
    schema = FunctionSchema(
        "data/input/functions_definition.json"
    )
    constraints = TokenConstraints(model, schema)
    print("Fonction actuelle :", constraints.current_function)

    constraints.update_current_function(
        [8822, 1889, 3744]
    )

    print("Fonction actuelle :", constraints.current_function)

    print(
        constraints.get_function_name_from_tokens(
            [8822, 1889, 3744]
        )
    )

    print(
        constraints.get_function_name_from_tokens(
            [999]
        )
    )

    print("Test process_token_text:")
    constraints.process_token_text('{"age":25}')

    print("\nTest JSONDecoder avec des tokens:")
    constraints.decoder = JSONDecoder()

    test_token_ids = [90, 314, 341, 515, 0, 92]

    for token_id in test_token_ids:
        token_text = model.decode([token_id])
        valid = constraints.decoder.consume_token(token_text)
        print(
            token_id,
            repr(token_text),
            "->",
            valid,
            constraints.decoder.state,
        )


if __name__ == "__main__":
    main()
