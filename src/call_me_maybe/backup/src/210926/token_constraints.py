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
        self.current_parameter = None

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

    # Create on 13:14 on 21/09/2026
    def get_allowed_value_start_tokens(self) -> set[int]:
        parameter_type = self.get_current_parameter_type()

        if parameter_type == "string":
            return {1}
        if parameter_type == "number":
            return self.get_allowed_number_start_tokens()
        return set()

    # Create on 13:30 on 21/09/2026
    def get_allowed_number_start_tokens(self) -> set[int]:
        return {
            12,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
        }

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

    # Create on 10:36 on 21/09/2026
    def update_current_function_from_key(
        self,
        key: str,
    ) -> None:
        if key in self.schema.get_function_names():
            self.current_function = key

    def update_current_function(
        self,
        generated_tokens: list[int],
    ) -> None:
        name = self.get_function_name_from_tokens(generated_tokens)
        if name is not None:
            self.current_function = name

    # Create on 10:36 on 21/09/2026
    def update_current_parameter(
        self,
        key: str,
    ) -> None:
        function = self.schema.get_function(
            self.current_function
        )

        if function is None:
            return

        if key in function.parameters:
            self.current_parameter = key

    # Create on 11:36 on 21/09/2026
    def get_current_parameter_type(self) -> str | None:
        if self.current_function is None:
            return None

        if self.current_parameter is None:
            return None

        function = self.schema.get_function(
            self.current_function
        )

        if function is None:
            return None

        parameter = function.parameters.get(
            self.current_parameter
        )

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

            elif previous_state == JSONState.KEY_CONTENT:
                if char == '"':
                    key = self.context.finish_key()
                    print("Completed key:", key)
                    if self.current_function is None:
                        self.update_current_function_from_key(key)
                    else:
                        self.update_current_parameter(key)
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
        # self.update_current_function(generated_tokens)
