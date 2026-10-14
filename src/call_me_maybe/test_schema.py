from llm_sdk import Small_LLM_Model
from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints
from call_me_maybe.parsing import parse_args


def main() -> None:
    model = Small_LLM_Model()
    args = parse_args()
    # schema = FunctionSchema("data/input/functions_definition.json")
    schema = FunctionSchema(args.functions_definition)

    constraints = TokenConstraints(model, schema)

    encoded = constraints.encode_function_names()
    for name, token_ids in encoded.items():
        print(f"{name}: {token_ids}")

    for name, token_ids in encoded.items():
        reference_ids = (
            model.encode(name)
            .squeeze(0)
            .tolist()
        )

        print(f"\nFonction : {name}")
        print("Tokenizer personnalisé :", token_ids)
        print("Modèle de référence    :", reference_ids)

        assert token_ids == reference_ids


if __name__ == "__main__":
    main()
