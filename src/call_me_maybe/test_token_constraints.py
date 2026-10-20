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


def test_number_exponent(
    constraints: TokenConstraints,
) -> None:
    """A valid JSON number may contain an exponent."""
    print("""==== A valid JSON number may contain an exponent. ====""")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    for char in "1e3":
        assert decoder.consume_char(char), (
            f"Number parser rejected {char!r}"
        )


def test_invalid_number_exponents(
    constraints: TokenConstraints,
) -> None:
    """Reject incomplete or malformed number exponents."""
    print("""==== Reject incomplete or malformed number exponents. ====""")
    invalid_numbers = ["1e", "1e+", "1e-", "1e3.4"]

    for number in invalid_numbers:
        decoder = constraints.decoder
        decoder.state = JSONState.EXPECT_VALUE_START
        decoder.object_stack = ["object"]
        decoder.escape_next = False
        decoder.unicode_escape_remaining = 0

        rejected = False
        for char in number:
            if not decoder.consume_char(char):
                rejected = True
                break

        if number == "1e3.4":
            assert rejected, f"Accepted invalid number {number!r}"


def test_number_completion_state(
    constraints: TokenConstraints,
) -> None:
    """A number must reach a valid ending state."""
    print("""==== A number must reach a valid ending state. ====""")
    decoder = constraints.decoder
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    decoder.state = JSONState.EXPECT_VALUE_START
    assert decoder.consume_char("1")
    assert decoder.state == JSONState.VALUE_NUMBER

    decoder.state = JSONState.EXPECT_EXPONENT
    assert decoder.consume_char("3")
    assert decoder.state == JSONState.EXPECT_EXPONENT_NUMBER


def test_number_exponent_sequences(
    constraints: TokenConstraints,
) -> None:
    """Test complete number exponent sequences."""
    print("""==== Test complete number exponent sequences. ====""")
    valid_numbers = ["1e3", "1E3", "1e+3", "1e-3", "0e3"]

    for number in valid_numbers:
        decoder = constraints.decoder
        decoder.state = JSONState.EXPECT_VALUE_START
        decoder.object_stack = ["object"]
        decoder.escape_next = False
        decoder.unicode_escape_remaining = 0

        for char in number:
            assert decoder.consume_char(char), (
                f"Rejected {char!r} in {number!r}"
            )


def test_reject_fraction_after_exponent(
    constraints: TokenConstraints,
) -> None:
    """A JSON number cannot have a fraction after its exponent."""
    print("==== A JSON number cannot have a fraction after its exponent. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    for char in "1e3":
        assert decoder.consume_char(char), (
            f"Unexpected rejection of {char!r}"
        )

    assert not decoder.consume_char("."), (
        "A decimal point must be rejected after an exponent"
    )


def test_fraction_with_exponent(
    constraints: TokenConstraints,
) -> None:
    """A fraction may be followed by a valid exponent."""
    print("=== A fraction may be followed by a valid exponent. ===")
    valid_numbers = ["1.5e3", "0.25E+2", "10.0e-1"]

    for number in valid_numbers:
        decoder = constraints.decoder
        decoder.state = JSONState.EXPECT_VALUE_START
        decoder.object_stack = ["object"]
        decoder.escape_next = False
        decoder.unicode_escape_remaining = 0

        for char in number:
            assert decoder.consume_char(char), (
                f"Rejected {char!r} in {number!r}"
            )


def test_fraction_requires_digit(
    constraints: TokenConstraints,
) -> None:
    """A decimal point must be followed by a digit."""
    print("==== A decimal point must be followed by a digit. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("1")
    assert decoder.consume_char(".")
    assert decoder.state == JSONState.EXPECT_FRACTION_DIGIT

    assert not decoder.consume_char("e")
    assert not decoder.consume_char("}")

    assert decoder.consume_char("5")
    assert decoder.state == JSONState.VALUE_NUMBER


def test_reject_leading_zero(
    constraints: TokenConstraints,
) -> None:
    """JSON numbers cannot have leading zeros."""
    print("==== JSON numbers cannot have leading zeros. ====")
    invalid_numbers = ["01", "00", "0123"]

    for number in invalid_numbers:
        decoder = constraints.decoder
        decoder.state = JSONState.EXPECT_VALUE_START
        decoder.object_stack = ["object"]
        decoder.escape_next = False
        decoder.unicode_escape_remaining = 0

        assert decoder.consume_char(number[0])
        assert not decoder.consume_char(number[1]), (
            f"Accepted a leading zero in {number!r}"
        )


def test_reject_negative_leading_zero(
    constraints: TokenConstraints,
) -> None:
    """Reject negative numbers with leading zeros."""
    print("==== Reject negative numbers with leading zeros. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("-")
    assert decoder.consume_char("0")
    assert not decoder.consume_char("1"), (
        "Accepted an invalid number: -01"
    )


def test_zero_followed_by_whitespace(
    constraints: TokenConstraints,
) -> None:
    """Allow whitespace after a zero value."""
    print("==== Allow whitespace after a zero value. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("0")
    assert decoder.consume_char(" ")
    assert decoder.state == JSONState.EXPECT_COMMA


def test_negative_zero_followed_by_whitespace(
    constraints: TokenConstraints,
) -> None:
    """Allow whitespace after negative zero."""
    print("==== Allow whitespace after negative zero. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("-")
    assert decoder.consume_char("0")
    assert decoder.consume_char(" ")
    assert decoder.state == JSONState.EXPECT_COMMA


def test_reject_number_continuation_after_whitespace(
    constraints: TokenConstraints,
) -> None:
    """Reject number continuation after whitespace."""
    print("==== Reject number continuation after whitespace. ====")
    invalid_sequences = ["0 1", "0 .5", "0 e3"]

    for sequence in invalid_sequences:
        decoder = constraints.decoder
        decoder.state = JSONState.EXPECT_VALUE_START
        decoder.object_stack = ["object"]
        decoder.escape_next = False
        decoder.unicode_escape_remaining = 0

        assert decoder.consume_char(sequence[0])
        assert decoder.consume_char(sequence[1])
        assert not decoder.consume_char(sequence[2]), (
            f"Accepted invalid sequence {sequence!r}"
        )


def test_zero_followed_by_comma(
    constraints: TokenConstraints,
) -> None:
    """Allow a comma after zero and require a new key."""
    print("==== Allow a comma after zero and require a new key. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("0")
    assert decoder.consume_char(",")
    assert decoder.state == JSONState.EXPECT_KEY_START
    assert not decoder.consume_char("}")


def test_whitespace_after_comma(
    constraints: TokenConstraints,
) -> None:
    """Allow whitespace between a comma and the next key."""
    print("==== Allow whitespace between a comma and the next key. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("0")
    assert decoder.consume_char(",")
    assert decoder.consume_char(" ")
    assert decoder.state == JSONState.EXPECT_KEY_START
    assert decoder.consume_char('"')
    assert decoder.state == JSONState.KEY_CONTENT


def test_reject_trailing_comma_with_whitespace(
    constraints: TokenConstraints,
) -> None:
    """Reject a closing brace after a comma and whitespace."""
    print("==== Reject a closing brace after a comma and whitespace. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("0")
    assert decoder.consume_char(",")
    assert decoder.consume_char(" ")
    assert not decoder.consume_char("}")


def test_reject_trailing_comma(
    constraints: TokenConstraints,
) -> None:
    """Reject a closing brace immediately after a comma."""
    print("==== Reject a closing brace immediately after a comma. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_VALUE_START
    decoder.object_stack = ["object"]
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    assert decoder.consume_char("0")
    assert decoder.consume_char(",")
    assert not decoder.consume_char("}")


def test_nested_json_object(
    constraints: TokenConstraints,
) -> None:
    """Accept a valid nested JSON object."""
    print("==== Accept a valid nested JSON object. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.START
    decoder.object_stack = []
    decoder.escape_next = False
    decoder.unicode_escape_remaining = 0

    json_text = '{"user": {"age": 20}}'

    for char in json_text:
        assert decoder.consume_char(char), (
            f"Rejected character {char!r} "
            f"in state {decoder.state}"
        )

    assert decoder.state == JSONState.DONE
    assert decoder.object_stack == []


def test_number_value_tokens(
    constraints: TokenConstraints,
) -> None:
    """Allow number-starting tokens for numeric parameters."""
    print("==== Allow number-starting tokens for numeric parameters. ====")
    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"
    constraints.decoder.state = JSONState.EXPECT_VALUE_START

    allowed = constraints.get_allowed_value_tokens()

    assert 12 in allowed, "The minus token must be allowed"

    for digit in "0123456789":
        token_ids = constraints.tokenizer.encode(digit)
        assert token_ids[0] in allowed, (
            f"Missing token for digit {digit}"
        )


def test_fraction_value_tokens(
    constraints: TokenConstraints,
) -> None:
    """Only digits may follow a decimal point."""
    print("==== Only digits may follow a decimal point. ====")
    decoder = constraints.decoder
    decoder.state = JSONState.EXPECT_FRACTION_DIGIT

    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"

    allowed = constraints.get_allowed_value_tokens()

    for digit in "0123456789":
        token_ids = constraints.tokenizer.encode(digit)
        assert token_ids[0] in allowed, (
            f"Missing token for digit {digit}"
        )

    forbidden_chars = ["e", "-", "}"]

    for char in forbidden_chars:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] not in allowed, (
            f"Unexpected token for {char!r}"
        )


def test_negative_number_value_tokens(
    constraints: TokenConstraints,
) -> None:
    """Only digits may follow a minus sign."""
    print("==== Only digits may follow a minus sign. ====")
    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"
    constraints.decoder.state = JSONState.EXPECT_NUMBER_DIGIT

    allowed = constraints.get_allowed_value_tokens()

    for digit in "0123456789":
        token_ids = constraints.tokenizer.encode(digit)
        assert token_ids[0] in allowed, (
            f"Missing token for digit {digit}"
        )

    for char in [".", "e", "}"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] not in allowed, (
            f"Unexpected token for {char!r}"
        )


def test_zero_value_tokens(
    constraints: TokenConstraints,
) -> None:
    """Check valid continuations after a zero."""
    print("==== Check valid continuations after a zero. ====")
    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"
    constraints.decoder.state = JSONState.EXPECT_ZERO_END

    allowed = constraints.get_allowed_value_tokens()

    for char in [".", "e"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] in allowed, (
            f"Missing token for {char!r}"
        )

    for char in ["1", "-"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] not in allowed, (
            f"Unexpected token for {char!r}"
        )


def test_integer_value_tokens(
    constraints: TokenConstraints,
) -> None:
    """Check valid continuations after an integer."""
    print("==== Check valid continuations after an integer. ====")
    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"
    constraints.decoder.state = JSONState.VALUE_NUMBER

    allowed = constraints.get_allowed_value_tokens()

    for char in ["0", "5", ".", "e", "E"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] in allowed, (
            f"Missing token for {char!r}"
        )

    for char in ["-", "}"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] not in allowed, (
            f"Unexpected token for {char!r}"
        )


def test_exponent_value_tokens(
    constraints: TokenConstraints,
) -> None:
    """Check valid tokens in exponent states."""
    print("==== Check valid tokens in exponent states. ====")
    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"

    constraints.decoder.state = JSONState.EXPECT_EXPONENT
    allowed = constraints.get_allowed_value_tokens()

    for char in ["+", "-", "3"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] in allowed, (
            f"Missing token for {char!r} after exponent marker"
        )

    constraints.decoder.state = JSONState.EXPECT_EXPONENT_DIGIT
    allowed = constraints.get_allowed_value_tokens()

    for char in ["0", "5"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] in allowed, (
            f"Missing token for {char!r} after exponent sign"
        )

    for char in ["+", "-", "."]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] not in allowed, (
            f"Unexpected token for {char!r} after exponent sign"
        )


def test_exponent_number_tokens(
    constraints: TokenConstraints,
) -> None:
    """Check valid continuations after a complete exponent."""
    print("==== Check valid continuations after a complete exponent. ====")
    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"
    constraints.decoder.state = JSONState.EXPECT_EXPONENT_NUMBER

    allowed = constraints.get_allowed_value_tokens()

    for digit in "0123456789":
        token_ids = constraints.tokenizer.encode(digit)
        assert token_ids[0] in allowed, (
            f"Missing token for digit {digit}"
        )

    for char in [".", "+", "-"]:
        token_ids = constraints.tokenizer.encode(char)
        assert token_ids[0] not in allowed, (
            f"Unexpected token for {char!r}"
        )


def test_process_function_name_token(
    constraints: TokenConstraints,
) -> None:
    """Process a token belonging to a function name."""
    print("==== Process a token belonging to a function name. ====")
    constraints.decoder.state = JSONState.VALUE_CONTENT
    constraints.current_key = "name"
    constraints.current_function = None
    constraints.function_name_tokens = []
    constraints.reading_function_name = True

    token_id = 8822  # Token representing 'fn'
    constraints.process_token(token_id, [token_id])

    assert constraints.function_name_tokens == [token_id]
    assert constraints.reading_function_name is True
    assert constraints.current_function is None


def test_complete_function_name(
    constraints: TokenConstraints,
) -> None:
    """Recognize a function after all its tokens are generated."""
    print("==== Recognize a function after all its tokens are generated. ====")
    name = "fn_greet"
    tokens = constraints.encode_function_names()[name]

    constraints.current_function = None
    constraints.function_name_tokens = []

    for token_id in tokens:
        constraints.function_name_tokens.append(token_id)
        constraints.update_current_function(
            constraints.function_name_tokens
        )

    assert constraints.current_function == name


def test_process_complete_function_name(
    constraints: TokenConstraints,
) -> None:
    """Recognize a function through process_token."""
    print("==== Recognize a function through process_token. ====")
    name = "fn_greet"
    tokens = constraints.encode_function_names()[name]

    constraints.decoder.state = JSONState.VALUE_CONTENT
    constraints.current_key = "name"
    constraints.current_function = None
    constraints.function_name_tokens = []
    constraints.reading_function_name = True

    for token_id in tokens:
        constraints.process_token(token_id, [])

    assert constraints.current_function == name


def test_get_current_parameter_type(
    constraints: TokenConstraints,
) -> None:
    """Return the type of the current function parameter."""
    print("==== Return the type of the current function parameter. ====")
    constraints.current_function = "fn_greet"
    constraints.current_parameter = "name"

    parameter_type = constraints.get_current_parameter_type()

    assert parameter_type == "string"


def test_allowed_tokens_for_string_parameter(
    constraints: TokenConstraints,
) -> None:
    """Allow JSON string-start tokens for a string parameter."""
    print("==== Allow JSON string-start tokens for a string parameter. ====")
    constraints.current_function = "fn_greet"
    constraints.current_parameter = "name"
    constraints.decoder.state = JSONState.EXPECT_VALUE_START

    constraints.build_string_start_tokens()

    allowed = constraints.get_allowed_value_tokens()

    assert allowed
    assert 1 in allowed


def test_allowed_tokens_for_number_parameter(
    constraints: TokenConstraints,
) -> None:
    """Allow number-starting tokens for a numeric parameter."""
    print("==== Allow number-starting tokens for a numeric parameter. ====")
    constraints.current_function = "fn_add_numbers"
    constraints.current_parameter = "a"
    constraints.decoder.state = JSONState.EXPECT_VALUE_START

    allowed = constraints.get_allowed_value_tokens()

    assert allowed
    assert 12 in allowed
    assert constraints.tokenizer.encode("5")[0] in allowed


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
    test_number_exponent(constraints)
    test_invalid_number_exponents(constraints)
    test_number_completion_state(constraints)
    test_number_exponent_sequences(constraints)
    test_reject_fraction_after_exponent(constraints)
    test_fraction_with_exponent(constraints)
    test_fraction_requires_digit(constraints)
    test_reject_leading_zero(constraints)
    test_reject_negative_leading_zero(constraints)
    test_zero_followed_by_whitespace(constraints)
    test_negative_zero_followed_by_whitespace(constraints)
    test_reject_number_continuation_after_whitespace(constraints)
    test_zero_followed_by_comma(constraints)
    test_reject_trailing_comma_with_whitespace(constraints)
    test_reject_trailing_comma(constraints)
    test_nested_json_object(constraints)
    test_number_value_tokens(constraints)
    test_fraction_value_tokens(constraints)
    test_negative_number_value_tokens(constraints)
    test_zero_value_tokens(constraints)
    test_integer_value_tokens(constraints)
    test_exponent_value_tokens(constraints)
    test_exponent_number_tokens(constraints)
    test_process_function_name_token(constraints)
    test_complete_function_name(constraints)
    test_process_complete_function_name(constraints)
    test_get_current_parameter_type(constraints)
    test_allowed_tokens_for_string_parameter(constraints)
    test_allowed_tokens_for_number_parameter(constraints)

    print("\nTous les tests sont terminés.")


if __name__ == "__main__":
    main()
