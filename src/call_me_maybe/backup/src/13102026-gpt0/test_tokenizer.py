#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_tokenizer.py                                    :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/10/13 14:29:47 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/13 14:45:03 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from call_me_maybe.tokenizer import Tokenizer
from llm_sdk import Small_LLM_Model


def inspect_tokenizer_files(model: Small_LLM_Model) -> None:
    """Inspect the tokenizer configuration and BPE merges."""
    tokenizer_path = model.get_path_to_tokenizer_file()
    merges_path = model.get_path_to_merges_file()

    with open(tokenizer_path, "r", encoding="utf-8") as file:
        tokenizer_data = json.load(file)

    print("\n=== Structure de tokenizer.json ===")
    print("Clés principales :", list(tokenizer_data.keys()))

    for section in (
        "normalizer",
        "pre_tokenizer",
        "model",
        "decoder",
        "post_processor",
    ):
        print(f"\n--- {section} ---")
        print(json.dumps(
            tokenizer_data.get(section),
            indent=2,
            ensure_ascii=False,
        )[:2500])

    print("\n=== Début de merges.txt ===")
    with open(merges_path, "r", encoding="utf-8") as file:
        for _ in range(10):
            line = file.readline()
            if not line:
                break
            print(line.rstrip())


def test_tokenizer_loading(model: Small_LLM_Model) -> None:
    """Test loading the tokenizer resources."""
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("\n=== Test du chargement du tokenizer ===")
    print("Nombre de tokens :", len(tokenizer.vocab))
    print("Nombre de règles BPE :", len(tokenizer.merges))
    print("Token ID 1 :", tokenizer.id_to_token[1])
    print("Token ID 17 :", tokenizer.id_to_token[17])
    print("Configuration BPE :", tokenizer.config["model"]["type"])

    assert tokenizer.id_to_token[1] == '"'
    assert tokenizer.id_to_token[17] == "2"
    assert tokenizer.config["model"]["type"] == "BPE"
    assert len(tokenizer.merges) > 0


def test_tokenizer_merges(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("\n=== Inspection des règles BPE ===")
    tokenizer.inspect_merges(limit=15)


def test_merge_ranks(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("\n=== Test des rangs BPE ===")

    assert tokenizer.merge_ranks[("Ġ", "Ġ")] == 0
    assert tokenizer.merge_ranks[("ĠĠ", "ĠĠ")] == 1
    assert tokenizer.merge_ranks[("i", "n")] == 2

    rank = tokenizer.merge_ranks.get(("x", "y"))
    print("Rang de la paire ('x', 'y') :", rank)
    unknown_pair = ("<token_inexistant_1>", "<token_inexistant_2>")

    assert tokenizer.merge_ranks.get(unknown_pair) is None

    print("Les rangs BPE sont corrects.")


def main():
    model = Small_LLM_Model()
    inspect_tokenizer_files(model)
    test_tokenizer_loading(model)
    test_tokenizer_merges(model)
    test_merge_ranks(model)


if __name__ == "__main__":
    main()
