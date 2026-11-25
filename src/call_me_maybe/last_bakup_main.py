#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:43:04 by andry-ha            #+#    #+#            #
#   Updated: 2026/11/25 16:47:45 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import time
import copy
import re
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from .function_schema import FunctionSchema
from .json_state import JSONState
# from .models import FunctionCallResult, FunctionCallTest,
from .models import (
    FunctionCallResult,
    TestInput,
    ParameterType,
)
from .parsing import parse_args
from .token_constraints import TokenConstraints


def load_prompts(path: str) -> list[TestInput]:
    """Load prompts from a JSON file."""
    the_path = Path(path)
    try:
        with the_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as e:
        raise ValueError(f"Function calling file not found: {the_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in function calling file: {the_path}"
        ) from e

    if not isinstance(data, list):
        raise ValueError(
            f"Function calling file must contain a JSON list: {the_path}"
        )

    try:
        return [TestInput.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(
            f"Invalid prompt data in function calling file: {the_path}: {e}"
        ) from e


def _get_tokens_by_text(
    constraints: TokenConstraints,
    target_text: str,
    strip: bool = False,
) -> set[int]:
    """Return all token IDs whose text matches exactly."""
    return constraints.get_tokens_by_text(
        target_text,
        strip=strip,
    )


def identify_function(
    model: Small_LLM_Model,
    prompt: str,
    constraints: TokenConstraints,
    functions_description: str,
    max_new_tokens: int = 30,
) -> str:
    """Identify a function using constrained token generation."""

    encoded_names = constraints.encoded_function_names

    if not encoded_names:
        raise ValueError("No function is available.")

    # Vérifier qu'aucun nom complet n'est le préfixe d'un autre.
    for name, token_ids in encoded_names.items():
        for other_name, other_ids in encoded_names.items():
            if (
                name != other_name
                and len(other_ids) > len(token_ids)
                and other_ids[:len(token_ids)] == token_ids
            ):
                raise ValueError(
                    f"Ambiguous function names: {name} and {other_name}"
                )

    # instruction = (
    #     "You are a function-calling assistant. "
    #     "Identify the function that best matches the user's request. "
    #     "Return ONLY the exact function name. "
    #     "Do not return arguments, explanations or punctuation.\n\n"
    #     f"Available functions:\n{functions_description}\n\n"
    #     f"User request: {prompt}\n"
    #     "Function name:"
    # )
    instruction = (
        "Select the function matching the request.\n"
        "Return only its exact name.\n\n"
        f"Functions:\n{functions_description}\n\n"
        f"Request: {prompt}\n"
        "Function:"
    )

    input_ids = constraints.tokenizer.encode(instruction)
    generated_tokens: list[int] = []

    # Mesures de performance
    model_time = 0.0
    constraints_time = 0.0
    tokens_generated = 0

    for _ in range(max_new_tokens):

        # Mesurer le temps d'inférence du modèle.
        start = time.perf_counter()
        logits = model.get_logits_from_input_ids(input_ids)
        inference_time = time.perf_counter() - start
        model_time += time.perf_counter() - start

        print(
            f"[DEBUG] Inférence token {len(generated_tokens) + 1} : "
            f"{inference_time:.4f} s "
            f"(entrée : {len(input_ids)} tokens)"
        )

        # Mesurer le temps de calcul des contraintes.
        start = time.perf_counter()

        if not generated_tokens:
            allowed = constraints.get_allowed_first_function_tokens()
        else:
            allowed = constraints.get_allowed_next_tokens(
                generated_tokens
            )

        allowed = {
            token_id
            for token_id in allowed
            if 0 <= token_id < len(logits)
        }

        constraints_time += time.perf_counter() - start

        if not allowed:
            raise ValueError(
                "No valid token available for function identification."
            )

        next_id = max(
            allowed,
            key=lambda token_id: logits[token_id],
        )

        generated_tokens.append(next_id)
        input_ids.append(next_id)
        tokens_generated += 1

        if constraints.is_complete_function_name(generated_tokens):
            function_name = constraints.get_function_name_from_tokens(
                generated_tokens
            )

            if function_name is None:
                raise ValueError("Unable to retrieve the function name.")

            print(f"\nFonction identifiée : {function_name}")
            print(f"Tokens générés : {tokens_generated}")
            print(f"Temps du modèle : {model_time:.4f} s")
            print(f"Temps des contraintes : {constraints_time:.4f} s")
            print(
                f"Temps total mesuré : "
                f"{model_time + constraints_time:.4f} s"
            )

            return function_name

    raise ValueError(
        "Function identification incomplete: "
        f"{len(generated_tokens)}/{max_new_tokens} tokens generated."
    )


def identify_function_and_parameters(
    model: Small_LLM_Model,
    prompt: str,
    constraints: TokenConstraints,
    functions_description: str,
    schema: FunctionSchema,
) -> tuple[str, list[tuple[str, ParameterType]]]:
    """Identify the function and retrieve its parameters."""

    function_name = identify_function(
        model,
        prompt,
        constraints,
        functions_description,
    )

    parameters = schema.get_function_parameters(function_name)

    return function_name, parameters


def prepare_parameter_context(
    constraints: TokenConstraints,
    function_name: str,
    parameter_name: str,
) -> None:
    """Prepare the JSON decoder for generating one parameter value."""

    constraints.reset()

    prefix = f'{{"parameters":{{"{parameter_name}":'

    prefix_tokens = constraints.tokenizer.encode(prefix)
    generated_tokens: list[int] = []

    for token_id in prefix_tokens:
        generated_tokens.append(token_id)
        constraints.process_token(token_id, generated_tokens)

    constraints.current_function = function_name

    if constraints.current_parameter != parameter_name:
        raise ValueError(
            f"Unable to prepare parameter: {parameter_name}"
        )

    if constraints.decoder.state != JSONState.EXPECT_VALUE_START:
        raise ValueError(
            "Unexpected decoder state: "
            f"{constraints.decoder.state}"
        )


NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
)


def extract_numeric_values(prompt: str) -> list[str]:
    """Extract numeric values from a user request, in order."""
    return NUMBER_PATTERN.findall(prompt)


def generate_number_parameter(
    model: Small_LLM_Model,
    prompt: str,
    function_name: str,
    parameter_name: str,
    constraints: TokenConstraints,
    function_description: str = "",
    parameter_description: str = "",
    max_new_tokens: int = 30,
    parameter_index: int = 0,
    parameter_count: int = 1,
) -> str:
    """Generate one numeric parameter value."""

    prepare_parameter_context(
        constraints,
        function_name,
        parameter_name,
    )

    prefix = f'{{"parameters":{{"{parameter_name}":'

    # instruction = (
    #     "You are a function-calling assistant. "
    #     "Extract the exact numeric value requested by the user. "
    #     "Preserve the sign of negative numbers. "
    #     "Do not change the numeric value. "
    #     "Do not calculate or transform it. "
    #     "Generate only the JSON numeric value.\n\n"

    #     "Important extraction rules:\n"
    #     f"- You are extracting numeric argument "
    #     f"{parameter_index + 1} out of {parameter_count}.\n"
    #     "- Extract the corresponding number from the user request "
    #     "according to its position among the numeric arguments.\n"
    #     "- Never reuse a number extracted for another parameter.\n\n"

    #     f"User request: {prompt}\n"
    #     f"Function: {function_name}\n"
    #     f"Parameter: {parameter_name}\n"
    #     "Parameter type: number\n"
    #     f"JSON: {prefix}"
    # )
    instruction = (
        "Extract exactly one numeric argument from the user's request.\n"
        "Read the request from beginning to end and identify each number "
        "expression, including numbers written in words.\n"
        "Map the first number expression to the first numeric parameter, "
        "the second to the second numeric parameter, and so on.\n"
        "For the requested parameter, return only the number at its "
        "corresponding position. Do not reuse an earlier argument unless "
        "the same value is explicitly repeated in the request.\n"
        "Convert number words to digits while preserving sign and decimal "
        "value. Do not calculate, round, or otherwise transform the value.\n"
        "Return only one valid JSON number.\n\n"
        f"Function: {function_name}\n"
        f"Function description: {function_description}\n"
        f"Parameter: {parameter_name}\n"
        f"Parameter description: {parameter_description}\n"
        f"Argument position among numeric parameters: {parameter_index + 1} "
        f"of {parameter_count}\n"
        f"User request: {prompt}\n"
        f"JSON: {prefix}"
    )

    input_ids = constraints.tokenizer.encode(instruction)
    generated_tokens: list[int] = []

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        allowed = constraints.get_allowed_value_tokens()

        print(
            f"[DEBUG] État JSON : {constraints.decoder.state}"
        )
        print(
            f"[DEBUG] Type paramètre : "
            f"{constraints.get_current_parameter_type()}"
        )
        print(
            f"[DEBUG] Tokens autorisés avant filtrage : {len(allowed)}"
        )
        # Ne pas autoriser un token qui dépasse le séparateur.
        # La virgule doit être générée comme un token indépendant.
        if constraints.decoder.state == JSONState.VALUE_NUMBER:
            allowed = {
                token_id
                for token_id in allowed
                if constraints.get_token_text(token_id) == ","
                or (
                    "," not in constraints.get_token_text(token_id)
                    and "}" not in constraints.get_token_text(token_id)
                )
            }

        allowed = {
            token_id
            for token_id in allowed
            if (
                0 <= token_id < len(logits)
                and constraints.is_compact_token_valid(token_id)
            )
        }

        print(
            f"[DEBUG] Tokens autorisés après filtrage : {len(allowed)}"
        )
        print(
            f"[DEBUG] Taille logits : {len(logits)}"
        )

        if not allowed:
            print("[DEBUG] Aucun token autorisé.")
            print(
                f"[DEBUG] État final : {constraints.decoder.state}"
            )
            print(
                f"[DEBUG] Fonction : {constraints.current_function}"
            )
            print(
                f"[DEBUG] Paramètre : {constraints.current_parameter}"
            )
            raise ValueError(
                "No valid token available for numeric parameter "
                f"{parameter_name!r}."
            )

        next_id = max(
            allowed,
            key=lambda token_id: logits[token_id],
        )

        generated_tokens.append(next_id)
        input_ids.append(next_id)

        token_text = constraints.get_token_text(next_id)

        print(
            f"[DEBUG] Token généré : {token_text!r}"
        )

        previous_state = constraints.decoder.state

        constraints.process_token(next_id, generated_tokens)

        print(
            f"[DEBUG] Transition : "
            f"{previous_state} -> {constraints.decoder.state}"
        )

        if constraints.decoder.state == JSONState.EXPECT_KEY_START:
            break

    if constraints.decoder.state != JSONState.EXPECT_KEY_START:
        raise ValueError(
            "Numeric parameter generation incomplete: "
            f"final state = {constraints.decoder.state}"
        )

    generated_text = constraints.tokenizer.decode(generated_tokens)

    if not generated_text.endswith(","):
        raise ValueError(
            f"Expected a comma delimiter, got: {generated_text!r}"
        )

    value = generated_text[:-1].strip()

    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid numeric parameter: {value!r}"
        ) from e

    if isinstance(parsed_value, bool) or not isinstance(
        parsed_value,
        (int, float),
    ):
        raise ValueError(f"Expected a number, got: {value!r}")

    return value


def generate_string_parameter(
    model: Small_LLM_Model,
    prompt: str,
    function_name: str,
    parameter_name: str,
    constraints: TokenConstraints,
    function_description: str = "",
    parameter_description: str = "",
    previous_parameters: dict[str, Any] | None = None,
    max_new_tokens: int = 60,
) -> str:
    """Generate one string parameter value."""

    prepare_parameter_context(
        constraints,
        function_name,
        parameter_name,
    )

    prefix = f'{{"parameters":{{"{parameter_name}":'

    # instruction = (
    #     "You are a function-calling assistant. "
    #     "Extract the exact text requested by the user. "
    #     "Preserve the original words, capitalization and punctuation. "
    #     "Remove quotation marks used only to delimit the text. "
    #     "Do not add explanations or additional text. "
    #     "Generate only the JSON string value.\n\n"
    #     f"User request: {prompt}\n"
    #     f"Function: {function_name}\n"
    #     f"Parameter: {parameter_name}\n"
    #     "Parameter type: string\n"
    #     f"JSON: {prefix}"
    # )
    instruction = (
        "Extract exactly the value required by this function parameter.\n"
        "Interpret the parameter using the function description, parameter "
        "description, user request, and previously extracted arguments.\n"
        "Use only information supported by the request; do not invent "
        "additional content.\n"
        "If this parameter is source text, copy the relevant text exactly, "
        "removing only quotation marks used to delimit it.\n"
        "If this parameter is a regular expression or pattern, produce the "
        "shortest pattern that matches each intended occurrence separately. "
        "Do not include surrounding text or repeat a pattern to account for "
        "multiple occurrences. Use regex syntax when the task requires it.\n"
        "If this parameter is a replacement value, return only the text "
        "that should replace each match, not one copy per occurrence.\n"
        "If it is another kind of structured value, return that value in "
        "the representation required by its description.\n"
        "Do not explain or add unrelated text. "
        "Return only one JSON string.\n\n"
        f"Function: {function_name}\n"
        f"Function description: {function_description}\n"
        f"Parameter: {parameter_name}\n"
        f"Parameter description: {parameter_description}\n"
        "Previously extracted parameters (JSON): "
        f"{json.dumps(previous_parameters or {}, ensure_ascii=False)}\n"
        f"User request: {prompt}\n"
        f"JSON: {prefix}"
    )

    input_ids = constraints.tokenizer.encode(instruction)
    generated_tokens: list[int] = []

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        state = constraints.decoder.state

        if state == JSONState.EXPECT_VALUE_START:
            candidates = {
                token_id
                for token_id in constraints.get_allowed_string_tokens()
                if constraints.get_token_text(token_id) == '"'
            }

        elif state == JSONState.VALUE_CONTENT:
            candidates = constraints.get_allowed_string_tokens()

        elif state == JSONState.EXPECT_COMMA:
            candidates = {
                token_id
                for token_id in constraints.vocab_token_ids
                if constraints.get_token_text(token_id) == ","
            }

        else:
            raise ValueError(
                f"Unexpected JSON state for string: {state}"
            )

        allowed: set[int] = set()

        for token_id in candidates:
            if not 0 <= token_id < len(logits):
                continue

            token_text = constraints.get_token_text(token_id)

            if not token_text:
                continue

            decoder_copy = copy.deepcopy(constraints.decoder)

            if not decoder_copy.consume_text(token_text):
                continue

            # La virgule termine la valeur et fait passer
            # le décodeur à EXPECT_KEY_START.
            if state == JSONState.EXPECT_COMMA:
                if (
                    token_text == ","
                    and decoder_copy.state == JSONState.EXPECT_KEY_START
                ):
                    allowed.add(token_id)

                continue

            # Pendant la génération de la chaîne,
            if decoder_copy.state == JSONState.EXPECT_COMMA:
                # Ne pas autoriser la fermeture immédiate
                # d'une chaîne vide.
                if state == JSONState.EXPECT_VALUE_START:
                    continue

                # Autoriser uniquement la fermeture de la chaîne.
                if '"' not in token_text:
                    continue

            elif decoder_copy.state == JSONState.VALUE_CONTENT:
                # Autoriser le contenu de la chaîne.
                pass

            else:
                continue

            allowed.add(token_id)
        print(f"[DEBUG] Candidats valides : {len(allowed)}")

        if not allowed:
            raise ValueError(
                "No valid token available for string parameter "
                f"{parameter_name!r}."
            )

        next_id = max(
            allowed,
            key=lambda token_id: logits[token_id],
        )

        print(
            f"[DEBUG CHOIX] "
            f"token={constraints.get_token_text(next_id)!r}, "
            f"état={state}"
        )
        generated_tokens.append(next_id)
        input_ids.append(next_id)

        constraints.process_token(
            next_id,
            generated_tokens,
        )

        if constraints.decoder.state == JSONState.EXPECT_KEY_START:
            break

    if constraints.decoder.state != JSONState.EXPECT_KEY_START:
        raise ValueError(
            "String parameter generation incomplete: "
            f"final state = {constraints.decoder.state}"
        )

    generated_text = constraints.tokenizer.decode(generated_tokens)

    if not generated_text.endswith(","):
        raise ValueError(
            f"Expected a comma delimiter, got: {generated_text!r}"
        )

    value_text = generated_text[:-1]

    try:
        value = json.loads(value_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid string parameter: {value_text!r}"
        ) from e

    if not isinstance(value, str):
        raise ValueError(
            f"Expected a string, got: {value!r}"
        )

    return value


def generate_parameters(
    model: Small_LLM_Model,
    prompt: str,
    function_name: str,
    parameters: list[tuple[str, ParameterType]],
    constraints: TokenConstraints,
    schema: FunctionSchema,
) -> dict[str, Any]:
    """Generate parameter values using the runtime-loaded function schema."""

    function_definition = schema.get_function(function_name)
    if function_definition is None:
        raise ValueError(f"Unknown function: {function_name}")

    function_description = function_definition.description
    generated_parameters: dict[str, Any] = {}

    numeric_parameter_count = sum(
        1
        for _, parameter_type in parameters
        if parameter_type in ("number", "int", "integer")
    )

    numeric_parameter_index = 0

    numeric_values = extract_numeric_values(prompt)

    use_direct_numeric_values = (
        len(numeric_values) == numeric_parameter_count
    )

    for parameter_name, parameter_type in parameters:
        print(f"\nParamètre : {parameter_name}")
        print(f"Type : {parameter_type}")

        if parameter_type == "string":
            value = generate_string_parameter(
                model,
                prompt,
                function_name,
                parameter_name,
                constraints,
                function_description=function_description,
                parameter_description=(
                    function_definition.parameters[parameter_name].description
                ),
                previous_parameters=generated_parameters,
            )

        elif parameter_type in ("number", "int", "integer"):

            if use_direct_numeric_values:
                # Optimisation : extraction directe sans appel au modèle.
                value = json.loads(
                    numeric_values[numeric_parameter_index]
                )

            else:
                # Repli : génération par le modèle lorsque les nombres
                # sont écrits en lettres ou ne sont pas détectés.
                generated_value = generate_number_parameter(
                    model,
                    prompt,
                    function_name,
                    parameter_name,
                    constraints,
                    function_description=function_description,
                    parameter_description=(
                        function_definition.parameters[
                            parameter_name
                        ].description
                    ),
                    parameter_index=numeric_parameter_index,
                    parameter_count=numeric_parameter_count,
                )

                value = json.loads(generated_value)

            numeric_parameter_index += 1

        else:
            raise ValueError(
                f"Unsupported parameter type: {parameter_type}"
            )

        generated_parameters[parameter_name] = value

        print(f"Valeur générée : {value!r}")

    return generated_parameters


def generate_text(
    model: Small_LLM_Model,
    prompt: str,
    constraints: TokenConstraints,
    max_new_tokens: int = 200,
) -> str:
    """Generate JSON while constraining tokens to valid choices."""
    input_ids = constraints.tokenizer.encode(prompt)
    generated: list[int] = []

    constraints.reset()

    logits_time = 0.0
    constraints_time = 0.0
    tokens_generated = 0

    for _ in range(max_new_tokens):
        start = time.perf_counter()
        logits = model.get_logits_from_input_ids(input_ids)
        logits_time += time.perf_counter() - start

        start = time.perf_counter()
        state = constraints.decoder.state
        allowed: set[int] = set()

        # Génération du séparateur ":".
        if state == JSONState.EXPECT_COLON:
            allowed = _get_tokens_by_text(constraints, ":")

        # Début de la valeur de "name".
        elif (
            state == JSONState.EXPECT_VALUE_START
            and constraints.current_key == "name"
        ):
            allowed = _get_tokens_by_text(constraints, '"')

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
                allowed.add(1)

        # Valeur textuelle d'un autre champ.
        elif (
            state == JSONState.VALUE_CONTENT
            and not constraints.reading_function_name
        ):
            allowed = constraints.get_allowed_string_tokens()

        # Début de l'objet "parameters".
        elif (
            state == JSONState.EXPECT_VALUE_START
            and constraints.current_key == "parameters"
        ):
            allowed = _get_tokens_by_text(constraints, "{", strip=True)

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
            if constraints.current_function is None:
                print(
                    "Erreur : fonction non identifiée.",
                    "État :", state,
                    "Clé :", constraints.current_key,
                )
                break
            allowed = constraints.get_allowed_value_tokens()

        # Clés de premier niveau : imposer "name", puis "parameters".
        elif (
            state in (JSONState.EXPECT_KEY_OR_END, JSONState.EXPECT_KEY_START)
            and not constraints.reading_parameters
        ):
            allowed = _get_tokens_by_text(constraints, '"')

        # Contenu des clés de premier niveau.
        elif (
            state == JSONState.KEY_CONTENT
            and not constraints.reading_parameters
        ):
            expected_key = (
                "name" if constraints.current_key == "" else "parameters"
            )
            partial_key = constraints.context.current_key
            allowed = set()

            for token_id, token_text in constraints.vocab_tokens:
                if token_text == '"' and partial_key == expected_key:
                    allowed.add(token_id)
                elif (
                    token_text
                    and token_text != '"'
                    and expected_key.startswith(partial_key + token_text)
                ):
                    allowed.add(token_id)

        # Clés de l'objet "parameters".
        elif (
            state in (JSONState.EXPECT_KEY_OR_END, JSONState.EXPECT_KEY_START)
            and constraints.reading_parameters
            and len(constraints.decoder.object_stack) == 2
        ):
            if constraints.current_function is not None:
                missing_parameters = constraints.get_missing_parameters()
                if missing_parameters:
                    allowed = _get_tokens_by_text(constraints, '"')
                elif state == JSONState.EXPECT_KEY_OR_END:
                    allowed = (
                        _get_tokens_by_text(constraints, "}", strip=True)
                    )

        # Contenu des clés de l'objet "parameters".
        elif (
            state == JSONState.KEY_CONTENT
            and constraints.reading_parameters
            and len(constraints.decoder.object_stack) == 2
        ):
            if constraints.current_function is not None:
                missing_parameters = constraints.get_missing_parameters()
                partial_key = constraints.context.current_key

                # DEBUG : vérifier pourquoi aucun token n'est autorisé
                print("\n--- DEBUG PARAMÈTRES ---")
                print("Fonction :", constraints.current_function)
                print("Clé partielle :", repr(partial_key))
                print("Paramètres utilisés :", constraints.used_parameters)
                print("Paramètres manquants :", missing_parameters)
                print("État JSON :", state)
                print("------------------------\n")

                allowed = set()

                for token_id, token_text in constraints.vocab_tokens:
                    if not token_text:
                        continue

                    if '"' not in token_text and any(
                        parameter.startswith(partial_key + token_text)
                        for parameter in missing_parameters
                    ):
                        allowed.add(token_id)

                    elif '"' in token_text:
                        key_part = token_text.split('"', 1)[0]

                        if partial_key + key_part in missing_parameters:
                            allowed.add(token_id)

        # Empêcher de fermer parameters si des champs obligatoires manquent.
        elif (
            state == JSONState.EXPECT_COMMA
            and constraints.reading_parameters
            and len(constraints.decoder.object_stack) == 2
        ):
            if constraints.current_function is not None:
                missing_parameters = constraints.get_missing_parameters()
                allowed = {
                    token_id
                    for token_id, token_text in constraints.vocab_tokens
                    if constraints.is_token_valid(token_id)
                    and not (
                        missing_parameters
                        and "}" in token_text
                    )
                }

        # Autres états du décodeur JSON.
        else:
            allowed = {
                token_id
                for token_id in constraints.vocab_token_ids
                if constraints.is_token_valid(token_id)
            }

        # Appliquer le mode JSON compact à toutes les branches.
        allowed = {
            token_id
            for token_id in allowed
            if constraints.is_compact_token_valid(token_id)
        }

        constraints_time += time.perf_counter() - start

        # Ne conserver que les identifiants présents dans les logits.
        allowed = {
            token_id for token_id in allowed if 0 <= token_id < len(logits)
        }

        if not allowed:
            print(
                "Aucun token autorisé.",
                "État :", state,
                "Clé :", constraints.current_key,
                "Fonction :", constraints.current_function,
            )
            break

        next_id = max(allowed, key=lambda token_id: logits[token_id])

        generated.append(next_id)
        input_ids.append(next_id)
        tokens_generated += 1

        # Consommer chaque token une seule fois.
        constraints.process_token(next_id, generated)

        if constraints.decoder.state == JSONState.DONE:
            break

    print(f"Tokens générés : {tokens_generated}")
    print(f"Temps du modèle : {logits_time:.2f} s")
    print(f"Temps des contraintes : {constraints_time:.2f} s")

    if constraints.decoder.state != JSONState.DONE:
        raise ValueError(
            "Génération JSON incomplète : "
            f"état final = {constraints.decoder.state}, "
            f"tokens générés = {tokens_generated}/{max_new_tokens}"
        )

    return constraints.tokenizer.decode(generated)


def main() -> None:
    """Load inputs and generate function calls using cached schemas."""
    args = parse_args()

    # Chargement des définitions et des prompts.
    schema = FunctionSchema(args.functions_definition)
    prompts = load_prompts(args.input)

    # Initialisation du modèle et des contraintes une seule fois.
    model = Small_LLM_Model()
    constraints = TokenConstraints(model, schema)

    # Description construite dynamiquement depuis functions_definition.json.
    functions_description = schema.get_functions_description()

    results: list[FunctionCallResult] = []

    print(f"Nombre de prompts à traiter : {len(prompts)}")

    for index, item in enumerate(prompts, start=1):
        prompt = item.prompt

        print("\n" + "=" * 60)
        print(f"Prompt {index}/{len(prompts)} : {prompt}")
        print("=" * 60)

        total_start = time.perf_counter()

        try:
            # 1. Identifier la fonction.
            constraints.reset()

            start = time.perf_counter()

            function_name, parameters = identify_function_and_parameters(
                model,
                prompt,
                constraints,
                functions_description,
                schema,
            )

            identification_time = time.perf_counter() - start

            print(f"\nFonction identifiée : {function_name}")
            print(f"Paramètres attendus : {parameters}")

            # 2. Générer uniquement les valeurs des paramètres.
            start = time.perf_counter()

            generated_parameters = generate_parameters(
                model,
                prompt,
                function_name,
                parameters,
                constraints,
                schema,
            )

            generation_time = time.perf_counter() - start

            # 3. Valider et normaliser les valeurs avec le schéma.
            normalized_parameters = schema.normalize_parameters(
                function_name,
                generated_parameters,
            )

            # 4. Construire le résultat final sans demander au LLM
            # de regénérer les noms de fonction et de paramètres.
            function_result = FunctionCallResult(
                prompt=prompt,
                name=function_name,
                parameters=normalized_parameters,
            )

            results.append(function_result)

            total_time = time.perf_counter() - total_start

            print("\nRésultat généré :")
            print(
                json.dumps(
                    function_result.model_dump(),
                    indent=4,
                    ensure_ascii=False,
                )
            )

            print("\nTemps d'exécution :")
            print(f"  Identification : {identification_time:.3f} s")
            print(f"  Génération     : {generation_time:.3f} s")
            print(f"  Total          : {total_time:.3f} s")

        except (ValueError, TypeError, KeyError) as error:
            print(f"\nErreur pour le prompt : {prompt!r}")
            print(f"Cause : {error}")
            raise

    # 5. Enregistrer les résultats.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [result.model_dump() for result in results],
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(f"Traitement terminé : {len(results)} prompts.")
    print(f"Résultats enregistrés dans : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
