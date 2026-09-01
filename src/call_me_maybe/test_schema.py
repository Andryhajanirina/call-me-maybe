from llm_sdk import Small_LLM_Model
from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints


def main() -> None:
    model = Small_LLM_Model()
    schema = FunctionSchema("data/input/functions_definition.json")

    constraints = TokenConstraints(model, schema)

    # print("Number of functions:", len(schema.functions))

    # for function in schema.functions:
    #     print()
    #     print("Function:", function.name)
    #     print("Description:", function.description)
    #     print("Parameters:")

    #     for name, parameter in function.parameters.items():
    #         print(f"  {name}: {parameter.type}")

    #     print("Returns:", function.returns.type)

    #     print(f">>>>>>>({schema.get_function_names()})")
    encoded = constraints.encode_function_names()
    for name, token_ids in encoded.items():
        print(f"{name}: {token_ids}")


if __name__ == "__main__":
    main()
