#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:43:04 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/25 14:00:02 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import time
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
    the_path = Path(path)
    try:
        with the_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return [TestInput.model_validate(item) for item in data]
    except FileNotFoundError as e:
        raise ValueError(
            f"Function calling file not found: {the_path}"
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in function calling file: {the_path}"
        ) from e


def generate_text(
    model: Small_LLM_Model,
    prompt: str,
    constraints: TokenConstraints,
    # max_new_tokens: int = 30,
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
    constraints.used_parameters = set()

    logits_time = 0.0
    constraints_time = 0.0
    tokens_generated = 0
    # for _ in range(max_new_tokens):
    #     logits = model.get_logits_from_input_ids(input_ids)
    #     allowed: set[int] = set()

    #     state = constraints.decoder.state
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        logits = model.get_logits_from_input_ids(input_ids)
        logits_time += time.perf_counter() - start

        allowed: set[int] = set()

        start = time.perf_counter()
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
            and constraints.reading_function_name
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
            and not constraints.reading_function_name
            # and constraints.current_key != "name"
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
                    missing_parameters = (
                        set(function.parameters)
                        - constraints.used_parameters
                    )

                    allowed = {
                        token_id
                        for token_id in constraints.vocab_token_ids
                        if constraints.get_token_text(token_id).strip() == '"'
                    }
                    if missing_parameters:
                        allowed = {
                            token_id
                            for token_id in allowed
                            if constraints.get_token_text(token_id).strip()
                            != "}"
                        }
        # Autres états du décodeur JSON.
        else:
            allowed = {
                token_id
                for token_id in constraints.vocab_token_ids
                if constraints.is_token_valid(token_id)
            }

        constraints_time += time.perf_counter() - start
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
        tokens_generated += 1

        # Le token est consommé ici une seule fois.
        constraints.process_token(next_id, generated)

        if constraints.decoder.state == JSONState.DONE:
            break
    # return cast(str, model.decode(generated))
    print(f"Tokens générés : {tokens_generated}")
    print(f"Temps du modèle : {logits_time:.2f} s")
    print(f"Temps des contraintes : {constraints_time:.2f} s")
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

    # prompt = prompts[0].prompt
    # prompt = prompts[3].prompt
    # instruction = (
    #     "You are a function-calling assistant. "
    #     "Available functions: fn_add_numbers(a, b), "
    #     "fn_greet(name), fn_reverse_string(s), "
    #     "fn_get_square_root(a), "
    #     "fn_substitute_string_with_regex("
    #     "source_string, regex, replacement). "
    #     "Return only a JSON object with keys "
    #     '"name" and "parameters". '
    #     f"User request: {prompt}\nJSON:"
    # )

    # # result = generate_text(model, instruction)
    # result = generate_text(model, instruction, constraints)
    # print("Prompt:", prompt)
    # print("Generated:", result)

    functions_description = schema.get_functions_description()
    for i in range(len(prompts)):
        prompt = prompts[i].prompt
        # instruction = (
        #     "You are a function-calling assistant. "
        #     f"Available functions:\n{functions_description}\n"
        #     "Return only a JSON object with keys "
        #     '"name" and "parameters". '
        #     f"User request: {prompt}\nJSON:"
        # )

        # instruction = (
        #     "You are a function-calling assistant. "
        #     "Choose the function that best matches the user's request. "
        #     "Return only one valid JSON object with exactly two "
        #     "top-level keys: name and parameters. "
        #     "The name field must contain the exact function name. "
        #     "The parameters field must contain the function arguments "
        #     "with their exact parameter names. "
        #     "For string parameters, preserve the requested text but "
        #     "remove quotation marks used only to delimit the text "
        #     "in the user's request. "
        #     "For fn_substitute_string_with_regex, generate a regex "
        #     "that matches the requested target and a replacement "
        #     "that exactly matches the requested replacement text. "
        #     "Use [0-9]+ to match sequences of digits. "
        #     "Use [aeiou] to match individual lowercase vowels. "
        #     "Use \\\\bcat\\\\b to match the whole word cat. "
        #     "Do not add unnecessary regex characters. "
        #     f"Available functions:\n{functions_description}\n"
        #     f"User request: {prompt}\nJSON:"
        # )
        instruction = (
            "You are a function-calling assistant. "
            "Choose the function that best matches the user's request. "
            "Return only one valid JSON object with exactly two "
            "top-level keys: name and parameters. "
            "The name field must contain the exact function name. "
            "The parameters field must contain the function arguments "
            "with their exact parameter names. "
            "For string parameters, preserve the requested text but "
            "remove quotation marks used only to delimit the text "
            "in the user's request. "
            "For fn_substitute_string_with_regex, generate a regex "
            "that matches the requested target and a replacement "
            "that exactly matches the requested replacement text. "
            "For regex functions, choose only the pattern needed "
            "to match the user's requested target. "
            "Do not combine unrelated patterns. "
            "Do not add unnecessary regex characters. "
            'Example: for the request "Reverse the string \'hello\'", '
            'return the parameter "s" with the value "hello". '
            f"Available functions:\n{functions_description}\n"
            f"User request: {prompt}\nJSON:"
        )

        # result = generate_text(model, instruction, constraints)
        start = time.perf_counter()

        result = generate_text(model, instruction, constraints)

        elapsed = time.perf_counter() - start
        print(f"Temps de génération : {elapsed:.2f} secondes")
        print("Prompt:", prompt)
        print("Generated:", result)


if __name__ == "__main__":
    main()
