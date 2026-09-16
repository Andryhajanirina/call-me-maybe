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

    print("Test process_token_text:")
    # constraints.process_token_text("age")
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
