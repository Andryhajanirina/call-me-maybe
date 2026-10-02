import json
from llm_sdk import Small_LLM_Model

from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints
# from call_me_maybe.json_decoder import JSONDecoder


def main() -> None:
    model = Small_LLM_Model()
    # vocab_path = model.get_path_to_vocab_file()

    # with open(vocab_path, "r", encoding="utf-8") as file:
    #     vocab = json.load(file)

    # print("Vocabulary type:", type(vocab))
    # print("Vocab size:", len(vocab))
    # # print("Vocab size:", len(model.vocab))

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

    # token_ids = [90, 1, 8822, 1889, 3744, 1]
    # token_ids = [1, 8822, 57507, 1]
    # token_ids = [90, 1, 8822, 57507, 1]

    print(
        '"fn_add_numbers" :',
        model.encode('"fn_add_numbers"').squeeze(0).tolist()
    )
    print(
        '"parameters" :',
        model.encode('"parameters"').squeeze(0).tolist()
    )
    # print('"a" :', model.encode('"a"').squeeze(0).tolist())
    # print('"b" :', model.encode('"b"').squeeze(0).tolist())
    # print('"2" :', model.encode('"2"').squeeze(0).tolist())
    # print('2 :', model.encode('2').squeeze(0).tolist())
    print('"fn_greet" :', model.encode('"fn_greet"').squeeze(0).tolist())
    print('"parameters" :', model.encode('"parameters"').squeeze(0).tolist())
    print('"name" :', model.encode('"name"').squeeze(0).tolist())
    print('"Andry" :', model.encode('"Andry"').squeeze(0).tolist())
    # print('"0" :', model.encode('"0"').squeeze(0).tolist())
    # print('0 :', model.encode('0').squeeze(0).tolist())
    # print('"1" :', model.encode('"1"').squeeze(0).tolist())
    # print('1 :', model.encode('1').squeeze(0).tolist())
    # print('"2" :', model.encode('"2"').squeeze(0).tolist())
    # print('2 :', model.encode('2').squeeze(0).tolist())
    # print('"9" :', model.encode('"9"').squeeze(0).tolist())
    # print('9 :', model.encode('9').squeeze(0).tolist())
    for digit in "0123456789":
        print(
            digit,
            ":",
            model.encode(digit).squeeze(0).tolist()
        )
    print('"-" :', model.encode('"9"').squeeze(0).tolist())
    print('- :', model.encode('9').squeeze(0).tolist())
    print("-", model.encode("-").squeeze(0).tolist())

    print("===========")
    # tokens = [
    #     90,                         # {
    #     31486, 1,                   # "name"
    #     25,                         # :
    #     1, 8822, 2891, 32964, 1,    # "fn_add_numbers"
    #     11,                         # ,
    #     1, 13786, 1,                # "parameters"
    #     25,                         # :
    #     90,                         # {
    #     56693, 1,                   # "a"
    #     25,                         # :
    #     17,                         # 2
    #     92,                         # }
    # ]
    # tokens = [
    #     90,                         # {
    #     31486, 1,                   # "name"
    #     25,                         # :
    #     1, 8822, 2891, 32964, 1,    # "fn_add_numbers"
    #     11,                         # ,
    #     1, 13786, 1,                # "parameters"
    #     25,                         # :
    #     90,                         # {
    #     56693, 1,                   # "a"
    #     25,                         # :
    #     17,                         # 2
    #     92,                         # }
    #     11,                         # ,
    #     90,                         # {
    #     65, 1,                      # "b"
    #     25,                         # :
    #     17,                         # 2
    #     92,                         # }
    #     92,                         # }
    # ]

    tokens = [
        90,                         # {
        31486, 1,                   # "name"
        25,                         # :
        1, 8822, 1889, 3744, 1,     # "fn_greet"
        11,                         # ,
        1, 13786, 1,                # "parameters"
        25,                         # :
        90,                         # {
        31486, 1,                   # "name"
        25,                         # :
        45916, 884, 1,           # "Andry"
        92,                         # }
        92,                         # }
    ]
    for token_id in tokens:
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
            "| Clé :",
            constraints.current_key,
            "| Paramètre :",
            constraints.current_parameter,
        )

    # function = schema.get_function(
    #     constraints.current_function
    # )

    # parameter = function.parameters[
    #     constraints.current_parameter
    # ]

    # print("Fonction :", function.name)
    # print("Fonction :", constraints.current_function)
    # print("Paramètre :", constraints.current_parameter)
    # print("Type :", parameter.type)
    print("Type :", constraints.get_current_parameter_type())
    print(
        "Allowed value tokens:",
        constraints.get_allowed_value_tokens()
    )

    logits = [0.0] * 30

    allowed = constraints.get_allowed_value_tokens()

    constrained = constraints.apply_token_constraint(
        logits,
        allowed,
    )

    print("Allowed:", allowed)
    print("State:", constraints.decoder.state)
    print("Token 17:", constrained[17])
    print("Token 20:", constrained[20])


if __name__ == "__main__":
    main()
