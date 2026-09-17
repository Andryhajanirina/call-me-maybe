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
        self.model = model
        self.schema = schema
        self.decoder = JSONDecoder()
        self.context = JSONContext()
        self.current_function = None

    def encode_function_names(self) -> dict[str, list[int]]:
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
        """
        Keep only allowed tokens and disable all others.
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
