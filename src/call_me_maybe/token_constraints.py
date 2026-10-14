import json
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
        self.current_function = None
        self.function_name_tokens: list[int] = []
        self.reading_function_name = False
        self.current_key = ""
        self.current_parameter = None
        self.reading_parameters = False
        self.string_start_tokens: set[int] = set()
        self.tokenizer = Tokenizer(
            vocab_path=model.get_path_to_vocab_file(),
            tokenizer_path=model.get_path_to_tokenizer_file(),
            merges_path=model.get_path_to_merges_file(),
        )
        self.token_text_cache: dict[int, str] = {}
        # self.string_content_tokens: set[int] | None = None

    def encode_function_names(self) -> dict[str, list[int]]:
        """Encode all available function names into token ID lists.

        Returns:
            Dictionary mapping function name to list of token IDs.
        """
        result: dict[str, list[int]] = {}

        for name in self.schema.get_function_names():
            # result[name] = self.model.encode(name).squeeze(0).tolist()
            result[name] = self.tokenizer.encode(name)

        return result

    def get_allowed_first_function_tokens(self) -> set[int]:
        """
        Return the token IDs that can start a function name.
        """
        encoded = self.encode_function_names()

        return {
            token_ids[0]
            for token_ids in encoded.values()
            if token_ids
        }

    def apply_token_constraint(
        self,
        logits: list[float],
        allowed_token_ids: set[int],
    ) -> list[float]:
        """Set logits of disallowed tokens to negative infinity.

        Args:
            logits: Original probability scores from LLM model.
            allowed_token_ids: Set of token IDs that preserve valid
                syntax/schema.

        Returns:
            Modified logits array.
        """
        constrained = [float("-inf")] * len(logits)

        for token_id in allowed_token_ids:
            constrained[token_id] = logits[token_id]

        return constrained

    def get_allowed_next_tokens(
        self,
        generated_tokens: list[int],
    ) -> set[int]:
        encoded = self.encode_function_names()

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
        for token_ids in self.encode_function_names().values():
            if token_ids == generated_tokens:
                return True
        return False

    def get_function_name_from_tokens(
        self,
        generated_tokens: list[int],
    ) -> str | None:
        encoded = self.encode_function_names()
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
        function = self.schema.get_function(
            self.current_function
        )
        parameter = function.parameters[
            self.current_parameter
        ]
        print("Current function:", function.name)
        print("Current parameter:", self.current_parameter)
        print("Type parameter:", parameter.type)
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

                    elif self.reading_parameters:
                        self.current_parameter = self.current_key

                    print(
                        "Completed key:",
                        self.current_key
                    )
                else:
                    self.context.add_key_character(char)

            print(
                repr(char),
                "->",
                valid,
                self.decoder.state,
            )

    def process_token(
        self,
        token_id: int,
        generated_tokens: list[int],
    ) -> None:
        token_text = self.get_token_text(token_id)

        self.process_token_text(token_text)

        if (
            self.current_key == "name"
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
        parameter_type = self.get_current_parameter_type()

        if parameter_type == "string":
            return self.get_allowed_string_tokens()

        if parameter_type == "number":
            allowed = {12}

            for digit in "0123456789":
                # token_ids = self.model.encode(digit).squeeze(0).tolist()
                token_ids = self.tokenizer.encode(digit)
                allowed.add(token_ids[0])

            return allowed

        return set()

    def starts_json_string(self, token_id: int) -> bool:
        token_text = self.get_token_text(token_id)
        return token_text.startswith('"')

    def build_string_start_tokens(self) -> None:
        vocab_path = self.model.get_path_to_vocab_file()

        with open(vocab_path, encoding="utf-8") as file:
            vocab = json.load(file)

        for token_id in vocab.values():
            token_text = self.get_token_text(token_id)
            if token_text.startswith('"'):
                self.string_start_tokens.add(token_id)

    def is_token_valid(self, token_id: int) -> bool:
        """Check whether a token preserves JSON syntax."""
        token_text = self.get_token_text(token_id)
        return self.decoder.consume_token(token_text)

    # def get_allowed_string_tokens(self) -> set[int]:
    #     """Return tokens compatible with a string parameter."""
    #     if self.decoder.state == JSONState.EXPECT_VALUE_START:
    #         return self.string_start_tokens.copy()

    #     if self.decoder.state == JSONState.VALUE_CONTENT:
    #         if self.string_content_tokens is None:
    #             allowed: set[int] = set()

    #             for token_id in range(self.model._tokenizer.vocab_size):
    #                 token_text = self.get_token_text(token_id)

    #                 if not token_text.startswith('"'):
    #                     if self.decoder.consume_token(token_text):
    #                         allowed.add(token_id)

    #             allowed.add(1)
    #             self.string_content_tokens = allowed

    #     return self.string_content_tokens.copy()

    def get_allowed_string_tokens(self) -> set[int]:
        """Return tokens compatible with a string parameter."""
        if self.decoder.state == JSONState.EXPECT_VALUE_START:
            return self.string_start_tokens.copy()

        if self.decoder.state == JSONState.VALUE_CONTENT:
            allowed: set[int] = set()

            for token_id in range(self.model._tokenizer.vocab_size):
                token_text = self.get_token_text(token_id)

                if not token_text.startswith('"'):
                    if self.decoder.consume_token(token_text):
                        allowed.add(token_id)

            allowed.add(1)
            return allowed

        return set()

    def get_token_text(self, token_id: int) -> str:
        """Return a token's decoded text, using the cache."""
        if token_id not in self.token_text_cache:
            self.token_text_cache[token_id] = (
                self.tokenizer.decode([token_id])
            )

        return self.token_text_cache[token_id]
