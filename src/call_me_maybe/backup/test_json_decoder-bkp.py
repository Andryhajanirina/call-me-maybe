from call_me_maybe.json_decoder import JSONDecoder
from call_me_maybe.json_state import JSONState


def test_initial_state() -> None:
    decoder = JSONDecoder()

    assert decoder.state == JSONState.START


def test_start_to_expect_key_start() -> None:
    decoder = JSONDecoder()

    assert decoder.consume_char("{") is True
    assert decoder.state == JSONState.EXPECT_KEY_START


def test_expect_key_start_to_key_content() -> None:
    decoder = JSONDecoder()

    decoder.consume_char("{")

    assert decoder.consume_char('"') is True
    assert decoder.state == JSONState.KEY_CONTENT


def test_key_content() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name')

    assert decoder.state == JSONState.KEY_CONTENT


def test_key_content_to_expect_colon() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name')

    assert decoder.consume_char('"') is True
    assert decoder.state == JSONState.EXPECT_COLON


def test_expect_colon_to_expect_value_start() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name"')

    assert decoder.consume_char(":") is True
    assert decoder.state == JSONState.EXPECT_VALUE_START


def test_expect_value_start_to_value_content() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name":')

    assert decoder.consume_char('"') is True
    assert decoder.state == JSONState.VALUE_CONTENT


def test_value_content() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name":"value')

    assert decoder.state == JSONState.VALUE_CONTENT


def test_value_content_to_expect_comma() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name":"value')

    assert decoder.consume_char('"') is True
    assert decoder.state == JSONState.EXPECT_COMMA


def test_expect_comma_to_expect_key_start() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name":"value"')

    assert decoder.state == JSONState.EXPECT_COMMA

    assert decoder.consume_char(",") is True
    assert decoder.state == JSONState.EXPECT_KEY_START


def test_multiple_key_value_pairs() -> None:
    decoder = JSONDecoder()

    text = '{"name":"value","city":"Paris"'

    assert decoder.consume_token(text) is True
    assert decoder.state == JSONState.EXPECT_COMMA


# def test_expect_comma_to_expect_object_end() -> None:
#     decoder = JSONDecoder()

#     decoder.consume_token('{"name":"value"')

#     assert decoder.state == JSONState.EXPECT_COMMA

#     assert decoder.consume_char("}") is True
#     assert decoder.state == JSONState.EXPECT_OBJECT_END

def test_expect_comma_to_done() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name":"value"')

    assert decoder.state == JSONState.EXPECT_COMMA

    assert decoder.consume_char("}") is True
    assert decoder.state == JSONState.DONE


def test_invalid_start() -> None:
    decoder = JSONDecoder()

    assert decoder.consume_char("[") is False
    assert decoder.state == JSONState.START


def test_invalid_key_start() -> None:
    decoder = JSONDecoder()

    decoder.consume_char("{")

    assert decoder.consume_char("a") is False
    assert decoder.state == JSONState.EXPECT_KEY_START


def test_invalid_colon() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name"')

    assert decoder.consume_char(",") is False
    assert decoder.state == JSONState.EXPECT_COLON


def test_invalid_value_start() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name":')

    assert decoder.consume_char("}") is False
    assert decoder.state == JSONState.EXPECT_VALUE_START


def test_invalid_comma() -> None:
    decoder = JSONDecoder()

    decoder.consume_token('{"name":"value"')

    assert decoder.consume_char(":") is False
    assert decoder.state == JSONState.EXPECT_COMMA


def test_consume_token_rollback() -> None:
    decoder = JSONDecoder()

    decoder.consume_char("{")

    assert decoder.state == JSONState.EXPECT_KEY_START

    # Le token commence correctement :
    # "name"
    #
    # mais "X" est invalide après la fermeture
    # de la clé car nous sommes alors dans EXPECT_COLON.
    assert decoder.consume_token('"name"X') is False

    # consume_token() doit restaurer l'état initial
    # du token en cas d'échec.
    assert decoder.state == JSONState.EXPECT_KEY_START


def test_object_end() -> None:
    decoder = JSONDecoder()

    assert decoder.consume_token("{") is True
    assert decoder.consume_token('"name"') is True
    assert decoder.consume_token(":") is True
    assert decoder.consume_token('"Andry"') is True

    assert decoder.state == JSONState.EXPECT_COMMA

    assert decoder.consume_token("}") is True
    assert decoder.state == JSONState.DONE
