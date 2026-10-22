#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:43:04 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/22 13:02:49 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from pathlib import Path
# from typing import cast

from llm_sdk import Small_LLM_Model

from .function_schema import FunctionSchema
from .parsing import parse_args
from .models import TestInput
from .token_constraints import TokenConstraints
from .json_decoder import JSONDecoder
from .json_context import JSONContext
from .json_state import JSONState
# from pydantic import ValidationError


def load_prompts(path: str) -> list[TestInput]:
    """Load prompts from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [TestInput.model_validate(item) for item in data]


def generate_text(
    model: Small_LLM_Model,
    prompt: str,
    constraints: TokenConstraints,
    max_new_tokens: int = 200,
) -> str:
    """Generate JSON while constraining tokens to valid choices."""
    input_ids = model.encode(prompt).tolist()[0]
    generated: list[int] = []

    # Réinitialiser l'état pour cette génération.
    constraints.decoder = JSONDecoder()
    constraints.context = JSONContext()
    constraints.current_function = None
    constraints.current_key = ""
    constraints.current_parameter = None
    constraints.function_name_tokens = []
    constraints.reading_function_name = False
    constraints.reading_parameters = False

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        allowed: set[int] = set()

        state = constraints.decoder.state

        # Début de la valeur de "name".
        if (
            state == JSONState.EXPECT_VALUE_START
            and constraints.current_key == "name"
        ):
            # allowed = constraints.get_allowed_string_tokens()
            allowed = {
                token_id
                for token_id in constraints.vocab_token_ids
                if constraints.get_token_text(token_id) == '"'
            }

        # Génération du nom de fonction.
        elif (
            state == JSONState.VALUE_CONTENT
            and constraints.current_key == "name"
        ):
            allowed = constraints.get_allowed_next_tokens(
                constraints.function_name_tokens
            )

            if constraints.is_complete_function_name(
                constraints.function_name_tokens
            ):
                allowed.add(1)  # ID du guillemet fermant, à vérifier.

        # Valeur textuelle d'un autre champ.
        elif (
            state == JSONState.VALUE_CONTENT
            and constraints.current_key != "name"
        ):
            allowed = constraints.get_allowed_string_tokens()

        # Début de l'objet "parameters".
        elif (
            state == JSONState.EXPECT_VALUE_START
            and constraints.current_key == "parameters"
        ):
            allowed = {
                token_id
                for token_id in constraints.vocab_token_ids
                if constraints.get_token_text(token_id).strip() == "{"
            }

        # Valeurs de paramètres et nombres.
        elif state in (
            JSONState.EXPECT_VALUE_START,
            JSONState.EXPECT_NUMBER_DIGIT,
            JSONState.EXPECT_ZERO_END,
            JSONState.VALUE_NUMBER,
            JSONState.EXPECT_FRACTION_DIGIT,
            JSONState.EXPECT_EXPONENT,
            JSONState.EXPECT_EXPONENT_DIGIT,
            JSONState.EXPECT_EXPONENT_NUMBER,
        ):
            # Cette branche suppose qu'une fonction et un paramètre
            # ont déjà été identifiés.
            if constraints.current_function is None:
                print(
                    "Erreur : fonction non identifiée.",
                    "État :", state,
                    "Clé :", constraints.current_key,
                )
                break

            allowed = constraints.get_allowed_value_tokens()

        # Clés de l'objet "parameters".
        elif (
            state in (
                JSONState.EXPECT_KEY_OR_END,
                JSONState.EXPECT_KEY_START,
            )
            and constraints.reading_parameters
        ):
            if constraints.current_function is None:
                allowed = set()
            else:
                function = constraints.schema.get_function(
                    constraints.current_function
                )

                if function is None:
                    allowed = set()
                else:
                    # Première étape : autoriser l'ouverture d'une clé.
                    allowed = {
                        token_id
                        for token_id in constraints.vocab_token_ids
                        if constraints.get_token_text(token_id).strip() == '"'
                    }
        # Autres états du décodeur JSON.
        else:
            allowed = {
                token_id
                for token_id in constraints.vocab_token_ids
                if constraints.is_token_valid(token_id)
            }

        # Filtrer les identifiants qui dépassent les logits.
        allowed = {
            token_id
            for token_id in allowed
            if 0 <= token_id < len(logits)
        }

        if not allowed:
            print(
                "Aucun token autorisé.",
                "État :", state,
                "Clé :", constraints.current_key,
                "Fonction :", constraints.current_function,
            )
            break

        constrained_logits = constraints.apply_token_constraint(
            logits,
            allowed,
        )

        next_id = max(
            allowed,
            key=lambda token_id: constrained_logits[token_id],
        )

        generated.append(next_id)
        input_ids.append(next_id)

        # Le token est consommé ici une seule fois.
        constraints.process_token(next_id, generated)

        if constraints.decoder.state == JSONState.DONE:
            break
    # return cast(str, model.decode(generated))
    return model.decode(generated)


def main() -> None:
    """Load inputs and initialize the language model."""
    args = parse_args()

    schema = FunctionSchema(args.functions_definition)
    prompts = load_prompts(args.input)

    model = Small_LLM_Model()

    constraints = TokenConstraints(model, schema)

    allowed = constraints.get_allowed_first_function_tokens()
    print("Tokens autorisés pour commencer un nom :", allowed)

    prompt = prompts[0].prompt
    instruction = (
        "You are a function-calling assistant. "
        "Available functions: fn_add_numbers(a, b), "
        "fn_greet(name), fn_reverse_string(s), "
        "fn_get_square_root(a), "
        "fn_substitute_string_with_regex("
        "source_string, regex, replacement). "
        "Return only a JSON object with keys "
        '"name" and "parameters". '
        f"User request: {prompt}\nJSON:"
    )

    # result = generate_text(model, instruction)
    result = generate_text(model, instruction, constraints)
    print("Prompt:", prompt)
    print("Generated:", result)


if __name__ == "__main__":
    main()
