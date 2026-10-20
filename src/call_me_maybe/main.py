#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:43:04 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/20 16:07:17 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore

from .function_schema import FunctionSchema
from .parsing import parse_args
from .models import TestInput
from .token_constraints import TokenConstraints
from .json_decoder import JSONDecoder
from .json_state import JSONState


def load_prompts(path: str) -> list[TestInput]:
    """Load prompts from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [TestInput.model_validate(item) for item in data]


def generate_text(
    model: Small_LLM_Model,
    prompt: str,
    max_new_tokens: int = 100,
) -> str:
    """Generate text and debug JSON decoding."""
    input_ids = model.encode(prompt).tolist()[0]
    generated: list[int] = []
    decoder = JSONDecoder()
    eos_token_id = model._tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = max(range(len(logits)), key=logits.__getitem__)

        if next_id == eos_token_id:
            break

        token_text = model.decode([next_id])
        if isinstance(token_text, list):
            token_text = "".join(token_text)

        print("Token ID:", next_id)
        print("Token text:", repr(token_text))
        print("Decoder state before:", decoder.state)

        valid = True
        for char in token_text:
            try:
                result = decoder.consume_char(char)
                print(
                    "Character:", repr(char),
                    "Valid:", result,
                    "State:", decoder.state,
                )
                if not result:
                    valid = False
                    break
            except Exception as error:
                print("Decoder exception:", repr(error))
                valid = False
                break

        if not valid:
            break

        generated.append(next_id)
        input_ids.append(next_id)

        if decoder.state == JSONState.DONE:
            break

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

    result = generate_text(model, instruction)
    print("Prompt:", prompt)
    print("Generated:", result)

    # print("Functions:", schema.get_function_names())
    # print("Prompts loaded:", len(prompts))
    # print("Model loaded:", model is not None)


if __name__ == "__main__":
    main()
