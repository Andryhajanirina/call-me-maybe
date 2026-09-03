from llm_sdk import Small_LLM_Model

from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints


def main() -> None:
    # 1. Initialisation du modèle
    model = Small_LLM_Model()

    # 2. Chargement du schema
    schema = FunctionSchema(
        "data/input/functions_definition.json"
    )

    # 3. Initialisation des contraintes
    constraints = TokenConstraints(
        model,
        schema,
    )

    # =========================test
    # print("After []:", constraints.get_allowed_next_tokens([]))

    # print("After [8822]:", constraints.get_allowed_next_tokens([8822]))

    # print(
    #     "After [8822, 2891]:",
    #     constraints.get_allowed_next_tokens([8822, 2891])
    # )
    prompt_ids = model.encode("What is the sum?").squeeze(0).tolist()
    generated_tokens = []
    for i in range(10):
        allowed = constraints.get_allowed_next_tokens(generated_tokens)

        if not allowed:
            print("No allowed tokens. Generation finished.")
            break

        input_ids = prompt_ids + generated_tokens
        logits = model.get_logits_from_input_ids(input_ids)
        constrained_logits = constraints.apply_token_constraint(
            logits,
            allowed,
        )

        finite_tokens = [
            token_id
            for token_id, logit in enumerate(constrained_logits)
            if logit != float("-inf")
        ]
        print("Number of allowed tokens:", len(finite_tokens))
        print("Allowed tokens:", finite_tokens)

        best_token_id = max(
            range(len(constrained_logits)),
            key=lambda token_id: constrained_logits[token_id],
        )

        print(
            "Best constrained token:",
            model.decode([best_token_id]),
        )
        generated_tokens.append(best_token_id)
        print("Generated tokens:", generated_tokens)
        print("Generated text:", model.decode(generated_tokens))

        print(
            constraints.is_complete_function_name(
                generated_tokens
            )
        )
    # =========================test


if __name__ == "__main__":
    main()
