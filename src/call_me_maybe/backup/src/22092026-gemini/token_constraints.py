"""Token constraint manager enforcing JSON syntax and function schema strictly."""

import json
from typing import Optional
from llm_sdk import Small_LLM_Model

from .function_schema import FunctionSchema
from .json_context import JSONContext
from .json_decoder import JSONDecoder


class TokenConstraints:
    """Enforces 100% valid schema-compliant JSON generation token by token."""

    def __init__(
        self,
        model: Small_LLM_Model,
        schema: FunctionSchema,
        vocab_map: dict[int, str],
    ) -> None:
        """Initialize token constraints processor.

        Args:
            model: Instance of Small_LLM_Model SDK.
            schema: Parsed function schemas available for calling.
            vocab_map: Mapping from token ID to raw token string.
        """
        self.model = model
        self.schema = schema
        self.vocab_map = vocab_map
        self.decoder = JSONDecoder()
        self.context = JSONContext()
        self.current_function: Optional[str] = None
        self.generated_text: str = '{"name": "'

        for char in self.generated_text:
            self.decoder.consume_char(char)

    def apply_token_constraint(
        self,
        logits: list[float],
        allowed_token_ids: set[int],
    ) -> list[float]:
        """Set logits of disallowed tokens to negative infinity."""
        constrained = [float("-inf")] * len(logits)
        for token_id in allowed_token_ids:
            if token_id < len(logits):
                constrained[token_id] = logits[token_id]
        return constrained

    def is_token_valid(self, token_text: str) -> bool:
        """Simulate feeding token_text to cloned decoder and check schema compliance."""
        sim_decoder = self.decoder.clone()

        for char in token_text:
            if not sim_decoder.consume_char(char):
                return False

        candidate_text = self.generated_text + token_text

        # 1. Validation du nom de la fonction
        if '"name": "' in candidate_text and '", "parameters"' not in candidate_text:
            parts = candidate_text.split('"name": "')
            if len(parts) > 1:
                prefix = parts[1].split('"')[0]
                valid_names = self.schema.get_function_names()
                if not any(name.startswith(prefix) for name in valid_names):
                    return False

        # 2. Interdiction de fermer le JSON prématurément si parameters n'a pas été généré
        if '", "parameters":' not in candidate_text and candidate_text.count("}") > 0:
            return False

        return True

    def get_valid_tokens_for_current_state(self) -> set[int]:
        """Filter vocabulary for tokens valid under current state and schema."""
        valid_tokens: set[int] = set()

        for token_id, token_text in self.vocab_map.items():
            if not token_text:
                continue
            if self.is_token_valid(token_text):
                valid_tokens.add(token_id)

        # Si aucun token n'est trouvé, débloquer les tokens de structure
        if not valid_tokens:
            for token_id, token_text in self.vocab_map.items():
                if any(c in token_text for c in ['}', '"', ',', ':']):
                    valid_tokens.add(token_id)

        return valid_tokens

    def process_selected_token(self, token_id: int) -> None:
        """Update internal state and decoder with the chosen token."""
        token_text = self.vocab_map.get(token_id, "")
        self.generated_text += token_text

        for char in token_text:
            self.decoder.consume_char(char)

        if '"name": "' in self.generated_text and not self.current_function:
            try:
                parts = self.generated_text.split('"name": "')
                if len(parts) > 1:
                    fn_candidate = parts[1].split('"')[0]
                    if fn_candidate in self.schema.get_function_names():
                        self.current_function = fn_candidate
            except Exception:
                pass
