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

    # 4. Récupération des tokens qui peuvent
    #    commencer un nom de fonction
    allowed = constraints.get_allowed_first_function_tokens()

    print("Allowed first tokens:", allowed)

    # 5. On demande au modèle ses logits
    input_ids = model.encode("What is the sum?")

    logits = model.get_logits_from_input_ids(
        input_ids.squeeze(0).tolist()
    )
    print("Number of logits:", len(logits))

    # 6. On applique notre contrainte
    constrained_logits = constraints.apply_token_constraint(
        logits,
        allowed,
    )

    # 6.1
    finite_tokens = [
        token_id
        for token_id, logit in enumerate(constrained_logits)
        if logit != float("-inf")
    ]

    print("Number of allowed tokens:", len(finite_tokens))
    print("Allowed tokens:", finite_tokens)

    # 7. On cherche le meilleur token après contrainte
    best_token_id = max(
        range(len(constrained_logits)),
        key=lambda token_id: constrained_logits[token_id],
    )

    print("Best constrained token ID:", best_token_id)

    print(
        "Best constrained token:",
        model.decode([best_token_id]),
    )


if __name__ == "__main__":
    main()
