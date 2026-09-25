from llm_sdk import Small_LLM_Model

from .function_schema import FunctionSchema
from .json_decoder import JSONDecoder
from .json_context import JSONContext
from .json_state import JSONState


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

    def encode_function_names(self) -> dict[str, list[int]]:
        """Encode all available function names into token ID lists.

        Returns:
            Dictionary mapping function name to list of token IDs.
        """
        result: dict[str, list[int]] = {}

        for name in self.schema.get_function_names():
            result[name] = self.model.encode(name).squeeze(0).tolist()

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
                    print(
                        "Completed key:",
                        self.context.finish_key(),
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
        token_text = self.model.decode([token_id])

        self.process_token_text(token_text)

        if self.decoder.state == JSONState.KEY_CONTENT:
            self.reading_function_name = True

        if self.reading_function_name:
            if token_text == '"':
                self.reading_function_name = False
            else:
                self.function_name_tokens.append(token_id)

        self.update_current_function(self.function_name_tokens)
