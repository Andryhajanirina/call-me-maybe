from llm_sdk import Small_LLM_Model

from .function_schema import FunctionSchema
from .json_decoder import JSONDecoder
from .json_context import JSONContext
from .json_state import JSONState
from call_me_maybe.tokenizer import Tokenizer


class TokenConstraints:
    def __init__(
        self,
        model: Small_LLM_Model,
        schema: FunctionSchema,
    ) -> None:
        """Initialize token constraints processor.

        Args:
            model: Instance of Small_LLM_Model SDK.
            schema: Parsed function schemas available for calling.
        """
        self.model = model
        self.schema = schema
        self.decoder = JSONDecoder()
        self.context = JSONContext()
        self.current_function: str | None = None
        self.function_name_tokens: list[int] = []
        self.reading_function_name = False
        self.current_key = ""
        self.current_parameter: str | None = None
        self.reading_parameters = False
        self.string_start_tokens: set[int] = set()
        self.tokenizer = Tokenizer(
            vocab_path=model.get_path_to_vocab_file(),
            tokenizer_path=model.get_path_to_tokenizer_file(),
            merges_path=model.get_path_to_merges_file(),
        )
        self.numeric_character_tokens = {
            char: self.tokenizer.encode(char)[0]
            for char in "-+0123456789.eE"
        }
        self.token_text_cache: dict[int, str] = {}
        self.vocab_token_ids: set[int] = set(
            self.tokenizer.id_to_token.keys()
        )

        self.vocab_tokens = [
            (token_id, self.get_token_text(token_id))
            for token_id in self.vocab_token_ids
        ]

        self.token_ids_by_text: dict[str, set[int]] = {}
        self.build_token_indexes()
        self.used_parameters: set[str] = set()
        self.encoded_function_names = self.encode_function_names()

    def reset(self) -> None:
        """Reset the state before a new generation."""
        self.decoder = JSONDecoder()
        self.context = JSONContext()
        self.current_function = None
        self.current_key = ""
        self.current_parameter = None
        self.function_name_tokens = []
        self.reading_function_name = False
        self.reading_parameters = False
        self.used_parameters = set()

    def get_character_tokens(self, characters: str) -> set[int]:
        """Return precomputed token IDs for the given characters."""
        return {
            self.numeric_character_tokens[char]
            for char in characters
        }

    def build_token_indexes(self) -> None:
        """Build indexes for frequently used token texts."""
        self.string_start_tokens.clear()
        self.token_ids_by_text.clear()

        for token_id, token_text in self.vocab_tokens:
            self.token_ids_by_text.setdefault(
                token_text, set()
            ).add(token_id)

            if token_text.startswith('"'):
                self.string_start_tokens.add(token_id)

    def get_tokens_by_text(
        self,
        target_text: str,
        strip: bool = False,
    ) -> set[int]:
        """Return token IDs matching the requested text."""
        if not strip:
            return self.token_ids_by_text.get(
                target_text, set()
            ).copy()

        return {
            token_id
            for token_id, token_text in self.vocab_tokens
            if token_text.strip() == target_text
        }

    def encode_function_names(self) -> dict[str, list[int]]:
        """Encode all available function names into token ID lists.

        Returns:
            Dictionary mapping function name to list of token IDs.
        """
        result: dict[str, list[int]] = {}

        for name in self.schema.get_function_names():
            result[name] = self.tokenizer.encode(name)

        return result

    def get_allowed_first_function_tokens(self) -> set[int]:
        """
        Return the token IDs that can start a function name.
        """
        encoded = self.encoded_function_names

        return {
            token_ids[0]
            for token_ids in encoded.values()
            if token_ids
        }

    def get_allowed_next_tokens(
        self,
        generated_tokens: list[int],
    ) -> set[int]:
        encoded = self.encoded_function_names

        allowed: set[int] = set()

        for token_ids in encoded.values():
            if token_ids[:len(generated_tokens)] == generated_tokens:
                if len(token_ids) > len(generated_tokens):
                    allowed.add(token_ids[len(generated_tokens)])

        return allowed

    def is_complete_function_name(
        self,
        generated_tokens: list[int],
    ) -> bool:
        for token_ids in self.encoded_function_names.values():
            if token_ids == generated_tokens:
                return True
        return False

    def get_function_name_from_tokens(
        self,
        generated_tokens: list[int],
    ) -> str | None:
        encoded = self.encoded_function_names
        for name, token_ids in encoded.items():
            if token_ids == generated_tokens:
                return name
        return None

    def update_current_function(
        self,
        generated_tokens: list[int],
    ) -> None:
        name = self.get_function_name_from_tokens(generated_tokens)
        if name is not None:
            self.current_function = name

    def get_current_parameter_type(self) -> str | None:
        if self.current_function is None or self.current_parameter is None:
            return None

        function = self.schema.get_function(self.current_function)

        if function is None:
            return None

        parameter = function.parameters.get(self.current_parameter)

        if parameter is None:
            return None

        return parameter.type

    def process_token_text(self, token_text: str) -> None:
        for char in token_text:
            previous_state = self.decoder.state

            valid = self.decoder.consume_char(char)

            if not valid:
                print(repr(char), "-> invalid")
                continue

            if previous_state in (
                JSONState.EXPECT_KEY_OR_END,
                JSONState.EXPECT_KEY_START,
            ) and char == '"':
                self.context.start_key()
                self.function_name_tokens = []
                self.reading_function_name = False

            elif previous_state == JSONState.KEY_CONTENT:
                if char == '"':
                    self.current_key = self.context.finish_key()

                    if self.current_key == "parameters":
                        self.reading_parameters = True
                        self.used_parameters = set()

                    elif self.reading_parameters:
                        self.current_parameter = self.current_key
                        self.used_parameters.add(self.current_key)

                    # print("Completed key:", self.current_key)
                else:
                    self.context.add_key_character(char)

            elif previous_state == JSONState.EXPECT_VALUE_START:
                if char == '"':
                    self.context.start_value()

            elif previous_state == JSONState.VALUE_CONTENT:
                if char != '"':
                    self.context.add_value_character(char)

    def process_token(
        self,
        token_id: int,
        generated_tokens: list[int],
    ) -> None:
        token_text = self.get_token_text(token_id)
        self.process_token_text(token_text)

        if (
            self.current_function is None
            and self.current_key == "name"
            and self.decoder.state == JSONState.VALUE_CONTENT
        ):
            self.reading_function_name = True

        if self.reading_function_name:
            if token_text == '"':
                self.reading_function_name = False
            else:
                self.function_name_tokens.append(token_id)

        self.update_current_function(self.function_name_tokens)

    def get_allowed_value_tokens(self) -> set[int]:
        """
        Determine and return the set of token IDs allowed by the
        current parameter type and JSON decoder state.

        Returns:
            set[int]: A set of allowed token IDs that satisfy both
                the current parameter type validation rules and the
                active JSON schema state. Returns an empty set if
                the parameter type is unsupported or if no
                transitions are valid.
        """
        parameter_type = self.get_current_parameter_type()

        if parameter_type == "string":
            return self.get_allowed_string_tokens()

        if parameter_type in ("int", "integer"):
            state = self.decoder.state

            # Beginning of an integer: allow minus or digits.
            if state == JSONState.EXPECT_VALUE_START:
                return self.get_character_tokens("-0123456789")

            # After a minus sign, at least one digit is required.
            if state == JSONState.EXPECT_NUMBER_DIGIT:
                return self.get_character_tokens("0123456789")

            # After a decimal point, at least one digit is required.
            if state == JSONState.EXPECT_FRACTION_DIGIT:
                return self.get_character_tokens("0123456789")

            # After zero or a complete integer, allow only
            # integer digits or valid JSON number terminators.
            if state in (
                JSONState.EXPECT_ZERO_END,
                JSONState.VALUE_NUMBER,
            ):
                allowed = set()
                valid_chars = "0123456789.,} \t\r\n"

                for token_id, token_text in self.vocab_tokens:

                    if not token_text:
                        continue

                    if not all(char in valid_chars for char in token_text):
                        continue

                    if self.decoder.is_token_valid(token_text):
                        allowed.add(token_id)

                return allowed

            return set()

        if parameter_type == "number":
            state = self.decoder.state

            # After a complete exponent, allow any token that preserves
            # valid JSON syntax, including the end of the number.
            if state == JSONState.EXPECT_EXPONENT_NUMBER:
                allowed = set()

                for token_id, token_text in self.vocab_tokens:

                    if (
                        token_text
                        and self.decoder.is_token_valid(token_text)
                    ):
                        allowed.add(token_id)

                return allowed

            digit_states = {
                JSONState.EXPECT_FRACTION_DIGIT,
                JSONState.EXPECT_EXPONENT_DIGIT,
            }

            if state in digit_states:
                return self.get_character_tokens("0123456789")

            if state == JSONState.EXPECT_ZERO_END:
                return self.get_character_tokens(".eE")

            if state == JSONState.VALUE_NUMBER:
                allowed = set()

                for token_id, token_text in self.vocab_tokens:

                    if not token_text or token_text == "}":
                        continue

                    if self.decoder.is_token_valid(token_text):
                        allowed.add(token_id)

                return allowed

            if state == JSONState.EXPECT_EXPONENT:
                return self.get_character_tokens("+-0123456789")

            allowed = {12}
            allowed.update(self.get_character_tokens("0123456789"))

            return allowed

        return set()

    def is_token_valid(self, token_id: int) -> bool:
        """Check whether a token preserves JSON syntax."""
        token_text = self.get_token_text(token_id)
        return self.decoder.is_token_valid(token_text)

    def is_compact_token_valid(self, token_id: int) -> bool:
        """Check whether a token is valid in compact JSON mode."""
        token_text = self.get_token_text(token_id)

        original_state = self.decoder.state
        original_stack = self.decoder.object_stack.copy()
        original_escape_next = self.decoder.escape_next
        original_unicode_escape_remaining = (
            self.decoder.unicode_escape_remaining
        )

        try:
            for char in token_text:
                # Refuser les espaces en dehors des chaînes JSON.
                if (
                    self.decoder.is_whitespace(char)
                    and self.decoder.state not in (
                        JSONState.KEY_CONTENT,
                        JSONState.VALUE_CONTENT,
                    )
                ):
                    return False

                if not self.decoder.consume_char(char):
                    return False

            return True

        finally:
            self.decoder.state = original_state
            self.decoder.object_stack = original_stack
            self.decoder.escape_next = original_escape_next
            self.decoder.unicode_escape_remaining = (
                original_unicode_escape_remaining
            )

    def get_allowed_string_tokens(self) -> set[int]:
        """Return tokens compatible with a string parameter."""
        if self.decoder.state == JSONState.EXPECT_VALUE_START:
            return self.string_start_tokens.copy()

        if self.decoder.state == JSONState.VALUE_CONTENT:
            allowed: set[int] = set()

            for token_id, token_text in self.vocab_tokens:

                if not token_text.startswith('"'):
                    if self.decoder.is_token_valid(token_text):
                        allowed.add(token_id)

            allowed.add(1)
            return allowed

        return set()

    def get_regex_character_count(self) -> int:
        """Return the number of characters generated for the regex value."""
        return len(self.context.finish_value())

    def get_token_text(self, token_id: int) -> str:
        """Return a token's decoded text, using the cache."""
        if token_id not in self.token_text_cache:
            self.token_text_cache[token_id] = (
                self.tokenizer.decode([token_id])
            )

        return self.token_text_cache[token_id]

    def get_missing_parameters(self) -> set[str]:
        """Return the required parameters not yet generated."""
        if self.current_function is None:
            return set()

        function = self.schema.get_function(self.current_function)

        if function is None:
            return set()

        return set(function.parameters) - self.used_parameters
