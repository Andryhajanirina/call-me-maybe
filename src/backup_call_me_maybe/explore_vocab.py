#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   explore_vocab.py                                     :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:43:34 by andry-ha            #+#    #+#            #
#   Updated: 2026/11/06 13:37:09 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json

from llm_sdk import Small_LLM_Model
from call_me_maybe.token_constraints import TokenConstraints
from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.parsing import parse_args


def main() -> None:
    model = Small_LLM_Model()
    args = parse_args()

    schema = FunctionSchema(args.functions_definition)
    constraints = TokenConstraints(model, schema)
    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    print("Vocabulary type:", type(vocab))
    print("Vocabulary size:", len(vocab))

    id_to_token = {token_id: token for token, token_id in vocab.items()}

    json_chars = ["{", "}", "[", "]", ":", ",", '"']

    for char in json_chars:
        print(f"\nTokens contenant {char!r}:")
        for token_id, token in id_to_token.items():
            if char in token and len(token) <= 4:
                print(token_id, repr(token))

    texts = [
        "{",
        '{"',
        '{"name"',
        '{"name":',
        '{"name":"',
        '{"name":"test"',
        '{"name":"test"}',
        '{"parameters":',
    ]

    for text in texts:
        ids = constraints.tokenizer.encode(text)
        print(repr(text), "->", ids)

        for token_id in ids:
            print("   ", token_id, repr(id_to_token.get(token_id)))


if __name__ == "__main__":
    main()
