#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_tokenizer.py                                    :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/10/13 14:29:47 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/14 13:24:43 by andry-ha           ###   ########.fr      #
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


def test_apply_bpe(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("\n=== Test de apply_bpe ===")

    result = tokenizer.apply_bpe(["i", "n"])

    print("Entrée : ['i', 'n']")
    print("Résultat :", result)

    assert result == ["in"]

    print("Fusion BPE simple réussie.")


def test_apply_bpe_multiple_merges(
    model: Small_LLM_Model,
) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("\n=== Test des fusions multiples ===")

    result = tokenizer.apply_bpe(
        ["e", "r", "i", "n"]
    )
    print("Entrée : ['e', 'r', 'i', 'n']")
    print("Résultat :", result)

    print("Rang e + r :", tokenizer.merge_ranks.get(("e", "r")))
    print("Rang i + n :", tokenizer.merge_ranks.get(("i", "n")))
    print("Rang er + in :", tokenizer.merge_ranks.get(("er", "in")))
    print("Rang in + in :", tokenizer.merge_ranks.get(("in", "in")))

    assert result == ["erin"]

    result = tokenizer.apply_bpe(["i", "n", "i", "n"])
    print("Entrée : ['i', 'n', 'i', 'n']")
    print("Résultat :", result)

    assert result == ["inin"]
    print("Fusions multiples réussies.")


def test_apply_bpe_does_not_modify_input(
    model: Small_LLM_Model,
) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    symbols = ["i", "n"]
    original = symbols.copy()

    tokenizer.apply_bpe(symbols)

    assert symbols == original
    print("La liste originale est préservée.")


def test_bytelevel_encoding(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("\n=== Test ByteLevel ===")

    assert tokenizer.encode_bytelevel("A") == "A"
    assert tokenizer.encode_bytelevel(" ") == "Ġ"
    assert tokenizer.encode_bytelevel("Hello world") == (
        "HelloĠworld"
    )

    encoded_accent = tokenizer.encode_bytelevel("é")
    print("Symbole ByteLevel pour é :", repr(encoded_accent))
    print("Nombre de symboles :", len(encoded_accent))

    assert len(encoded_accent) == 2
    print("Tests ByteLevel réussis.")


def test_bytelevel_round_trip(
    model: Small_LLM_Model,
) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    samples = [
        "Hello world",
        "é",
        "Antananarivo",
        "Bonjour !",
        "🙂",
    ]

    print("\n=== Test aller-retour ByteLevel ===")

    for original in samples:
        encoded = tokenizer.encode_bytelevel(original)
        decoded = tokenizer.decode_bytelevel(encoded)

        print(
            f"{original!r} -> {encoded!r} -> {decoded!r}"
        )
        assert decoded == original

    print("Tous les tests aller-retour sont réussis.")


def test_pre_tokenize(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    samples = [
        "Hello world!",
        "I'm learning Python.",
        "12345",
        "Bonjour !",
        "Antananarivo",
        "Hello\nworld",
        "🙂",
        "What is the sum of 2 and 3?",
    ]

    print("\n=== Test de pré-tokenisation ===")
    for sample in samples:
        pieces = tokenizer.pre_tokenize(sample)
        print(f"{sample!r} -> {pieces!r}")
        assert "".join(pieces) == sample

    print("Tous les tests de pré-tokenisation sont réussis.")


def test_tokenize_to_pieces(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    samples = [
        "Hello world!",
        "Bonjour !",
        "What is the sum of 2 and 3?",
        "I'm learning Python.",
        "é 🙂",
        "Replace all numbers in Hello 34 with NUMBERS",
    ]

    print("\n=== Test pré-tokenisation + ByteLevel + BPE ===")
    for sample in samples:
        pieces = tokenizer.tokenize_to_pieces(sample)
        print(f"{sample!r} -> {pieces!r}")

        reconstructed = "".join(
            tokenizer.decode_bytelevel("".join(pieces))
        )
        assert reconstructed == sample

    print("Tous les tests de tokenisation sont réussis.")


def test_encode(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    samples = [
        "Hello world!",
        "Bonjour !",
        "I'm learning Python.",
        "é 🙂",
        "What is the sum of 2 and 3?",
    ]

    print("\n=== Test encode ===")
    for sample in samples:
        token_ids = tokenizer.encode(sample)
        print(f"{sample!r} -> {token_ids!r}")

        assert token_ids
        assert all(isinstance(token_id, int) for token_id in token_ids)

    print("Tous les tests encode sont réussis.")


def test_encode_against_model(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    samples = [
        "Hello",
        "Hello world!",
        "Bonjour !",
        "I'm learning Python.",
        "12345",
        "é 🙂",
        "What is the sum of 2 and 3?",
        'Replace all numbers in "Hello 34" with NUMBERS',
        "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    ]

    print("\n=== Comparaison avec le tokenizer de référence ===")

    for sample in samples:
        expected = model.encode(sample).squeeze(0).tolist()
        actual = tokenizer.encode(sample)

        print(f"\nTexte : {sample!r}")
        print(f"Référence : {expected}")
        print(f"Notre code : {actual}")

        if expected == actual:
            print("Résultat : IDENTIQUE")
        else:
            print("Résultat : DIFFÉRENT")

    print("\nComparaison terminée.")


def test_encode_decode(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    samples = [
        "Hello world!",
        "Bonjour !",
        "I'm learning Python.",
        "12345",
        "é 🙂",
        "Antananarivo",
        "What is the sum of 2 and 3?",
        'Replace all numbers in "Hello 34" with NUMBERS',
        "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    ]

    print("\n=== Test encode/decode ===")

    for sample in samples:
        token_ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(token_ids)

        print(f"Original : {sample!r}")
        print(f"IDs      : {token_ids}")
        print(f"Décodé   : {decoded!r}")

        assert decoded == sample

    print("Tous les tests encode/decode sont réussis.")


def test_decode_against_model(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    samples = [
        "Hello world!",
        "Bonjour !",
        "I'm learning Python.",
        "12345",
        "é 🙂",
        "Antananarivo",
        'Replace all numbers in "Hello 34" with NUMBERS',
    ]

    print("\n=== Comparaison du décodage ===")

    for sample in samples:
        token_ids = tokenizer.encode(sample)

        expected = model.decode(token_ids)
        actual = tokenizer.decode(token_ids)

        print(f"\nTexte : {sample!r}")
        print(f"Référence : {expected!r}")
        print(f"Notre code : {actual!r}")

        assert actual == expected

    print("\nTous les tests de comparaison sont réussis.")


def test_decode_unknown_id(model: Small_LLM_Model) -> None:
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("\n=== Test ID inconnu ===")

    try:
        tokenizer.decode([-1])
    except ValueError as error:
        print(f"Erreur attendue : {error}")
    else:
        raise AssertionError("Un ID inconnu aurait dû provoquer une erreur")

    print("Test ID inconnu réussi.")


def main():
    model = Small_LLM_Model()
    # inspect_tokenizer_files(model)
    # test_tokenizer_loading(model)
    # test_tokenizer_merges(model)
    # test_merge_ranks(model)
    # test_apply_bpe(model)
    # test_apply_bpe_multiple_merges(model)
    # test_apply_bpe_does_not_modify_input(model)
    # test_bytelevel_encoding(model)
    # test_bytelevel_round_trip(model)
    # test_pre_tokenize(model)
    # test_tokenize_to_pieces(model)
    # test_encode(model)
    # test_encode_against_model(model)
    test_encode_decode(model)
    test_decode_against_model(model)
    test_decode_unknown_id(model)


if __name__ == "__main__":
    main()
