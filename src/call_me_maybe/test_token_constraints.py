import json
import time
from llm_sdk import Small_LLM_Model

from call_me_maybe.function_schema import FunctionSchema
from call_me_maybe.token_constraints import TokenConstraints
from call_me_maybe.json_state import JSONState
from call_me_maybe.tokenizer import Tokenizer


def test_string_start_tokens(
    model: Small_LLM_Model,
    constraints: TokenConstraints,
) -> None:
    """Test allowed tokens at the beginning of a string."""
    constraints.current_function = "fn_greet"
    constraints.current_parameter = "name"
    constraints.decoder.state = JSONState.EXPECT_VALUE_START
    constraints.decoder.object_stack = ["object"]

    allowed = constraints.get_allowed_value_tokens()

    print("\n=== Début de chaîne ===")
    print("Guillemet :", 1 in allowed)
    print('"A :', 29133 in allowed)
    print('"And :', 45916 in allowed)
    print("Nombre :", 17 in allowed)
    print("Objet :", 90 in allowed)

    assert 1 in allowed
    assert 29133 in allowed
    assert 45916 in allowed
    assert 17 not in allowed
    assert 90 not in allowed


def test_string_content(
    constraints: TokenConstraints,
) -> None:
    """Test allowed tokens inside a string."""
    constraints.decoder.state = JSONState.VALUE_CONTENT

    allowed = constraints.get_allowed_value_tokens()

    print("\n=== Contenu de chaîne ===")
    print("ry :", 884 in allowed)
    print("And :", 3036 in allowed)
    print("Guillemet :", 1 in allowed)

    assert 884 in allowed
    assert 3036 in allowed
    assert 1 in allowed


def test_apply_token_constraint(
    model: Small_LLM_Model,
    constraints: TokenConstraints,
) -> None:
    """Test that disallowed token logits become negative infinity."""
    constraints.current_function = "fn_greet"
    constraints.current_parameter = "name"
    constraints.decoder.state = JSONState.EXPECT_VALUE_START
    constraints.decoder.object_stack = ["object"]

    allowed = constraints.get_allowed_value_tokens()

    test_token_ids = [1, 17, 90, 29133, 45916]
    logits = [0.0] * model._tokenizer.vocab_size

    logits[1] = 1.0
    logits[17] = 10.0
    logits[90] = 8.0
    logits[29133] = 7.0
    logits[45916] = 6.0

    constrained = constraints.apply_token_constraint(
        logits,
        allowed,
    )

    print("\n=== Application des contraintes ===")

    for token_id in test_token_ids:
        print(
            token_id,
            repr(model.decode([token_id])),
            "| Autorisé :", token_id in allowed,
            "| Logit :", constrained[token_id],
        )

    for token_id in allowed:
        assert constrained[token_id] == logits[token_id]

    for token_id in test_token_ids:
        if token_id not in allowed:
            assert constrained[token_id] == float("-inf")

    assert constrained[1] == 1.0
    assert constrained[17] == float("-inf")
    assert constrained[90] == float("-inf")
    assert constrained[29133] == 7.0
    assert constrained[45916] == 6.0

    selected_token_id = max(
        test_token_ids,
        key=lambda token_id: constrained[token_id],
    )

    print(
        "Token sélectionné :",
        selected_token_id,
        repr(model.decode([selected_token_id])),
    )

    assert selected_token_id == 29133


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


def test_token_text_cache(
    constraints: TokenConstraints,
) -> None:
    """Test that decoded token texts are cached."""
    token_id = 29133

    constraints.token_text_cache.pop(token_id, None)

    first_text = constraints.get_token_text(token_id)
    cache_size = len(constraints.token_text_cache)

    second_text = constraints.get_token_text(token_id)

    assert first_text == second_text
    assert len(constraints.token_text_cache) == cache_size
    assert token_id in constraints.token_text_cache

    print("\n=== Test du cache des tokens ===")
    print("Texte :", repr(first_text))
    print("Taille du cache après le premier appel :", cache_size)
    print("Taille du cache après le second appel :", len(
        constraints.token_text_cache
    ))
    print("Cache fonctionnel.")


def test_string_content_timing(
    constraints: TokenConstraints,
) -> None:
    """Measure string content token filtering."""
    constraints.decoder.state = JSONState.VALUE_CONTENT
    constraints.decoder.object_stack = ["object"]

    start = time.perf_counter()
    allowed = constraints.get_allowed_string_tokens()
    elapsed = time.perf_counter() - start

    print(f"\nString content tokens: {len(allowed)}")
    print(f"Filtering duration: {elapsed:.3f} seconds")


def test_escaped_quote_in_string(
    constraints: TokenConstraints,
) -> None:
    """An escaped quote must not close a JSON string."""
    print("===== An escaped quote must not close a JSON string. =====")
    decoder = constraints.decoder
    decoder.state = JSONState.VALUE_CONTENT
    decoder.object_stack = ["object"]

    assert decoder.consume_char("\\")
    assert decoder.consume_char('"')
    assert decoder.state == JSONState.VALUE_CONTENT


def test_consume_token_restores_escape_state(
    constraints: TokenConstraints,
) -> None:
    """Token simulation must restore the escape state."""
    print("==== Token simulation must restore the escape state. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.VALUE_CONTENT
    decoder.object_stack = ["object"]
    decoder.escape_next = False

    assert decoder.consume_token("\\")
    assert decoder.escape_next is False
    assert decoder.state == JSONState.VALUE_CONTENT


def test_unicode_escape_in_string(
    constraints: TokenConstraints,
) -> None:
    """A Unicode escape must remain inside the string."""
    print("""==== A Unicode escape must remain inside the string. ====""")
    decoder = constraints.decoder
    decoder.state = JSONState.VALUE_CONTENT
    decoder.object_stack = ["object"]
    decoder.escape_next = False

    for char in '\\u0041':
        assert decoder.consume_char(char)

    assert decoder.state == JSONState.VALUE_CONTENT
    assert decoder.escape_next is False


def test_invalid_unicode_escape(
    constraints: TokenConstraints,
) -> None:
    """Reject a non-hex digit in a Unicode escape."""
    print("""==== Reject a non-hex digit in a Unicode escape. ====""")
    decoder = constraints.decoder
    decoder.state = JSONState.VALUE_CONTENT
    decoder.object_stack = ["object"]
    decoder.escape_next = False

    for char in '\\u00':
        assert decoder.consume_char(char)
    assert not decoder.consume_char("G")


def test_consume_token_restores_unicode_state(
    constraints: TokenConstraints,
) -> None:
    """Token simulation must restore Unicode escape state."""
    print("""==== Token simulation must restore Unicode escape state. ====""")
    decoder = constraints.decoder
    decoder.state = JSONState.VALUE_CONTENT
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_token("\\u")
    assert decoder.unicode_escape_remaining == 0
    assert decoder.escape_next is False


def test_escaped_quote_in_key(
    constraints: TokenConstraints,
) -> None:
    """An escaped quote must not close a JSON key."""
    print("""==== An escaped quote must not close a JSON key. ====""")
    decoder = constraints.decoder
    decoder.state = JSONState.KEY_CONTENT
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("\\")
    assert decoder.consume_char('"')
    assert decoder.state == JSONState.KEY_CONTENT


def main() -> None:
    """Run token constraint tests."""
    model = Small_LLM_Model()
    # inspect_tokenizer_files(model)
    test_tokenizer_loading(model)
    test_tokenizer_merges(model)
    test_merge_ranks(model)
    tokenizer = Tokenizer(
        vocab_path=model.get_path_to_vocab_file(),
        tokenizer_path=model.get_path_to_tokenizer_file(),
        merges_path=model.get_path_to_merges_file(),
    )

    print("Type du vocabulaire :", type(tokenizer.vocab))
    print("Nombre d'entrées :", len(tokenizer.vocab))

    schema = FunctionSchema(
        "data/input/functions_definition.json"
    )

    constraints = TokenConstraints(model, schema)
    constraints.build_string_start_tokens()

    test_string_start_tokens(model, constraints)
    test_string_content(constraints)
    test_apply_token_constraint(model, constraints)

    test_token_text_cache(constraints)

    test_string_content_timing(constraints)
    test_escaped_quote_in_string(constraints)
    test_consume_token_restores_escape_state(constraints)
    test_unicode_escape_in_string(constraints)
    test_invalid_unicode_escape(constraints)
    test_consume_token_restores_unicode_state(constraints)
    test_escaped_quote_in_key(constraints)

    print("\nTous les tests sont terminés.")


if __name__ == "__main__":
    main()
