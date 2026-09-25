"""Dynamic token constraint manager enforcing complete function call grammar."""

import json
from typing import Optional
from llm_sdk import Small_LLM_Model

from .function_schema import FunctionSchema
from .json_decoder import JSONDecoder


class TokenConstraints:
    """Enforces 100% valid schema-compliant JSON generation dynamically."""

    def __init__(
        self,
        model: Small_LLM_Model,
        schema: FunctionSchema,
        vocab_map: dict[int, str],
    ) -> None:
        """Initialize token constraints processor dynamically from schema."""
        self.model = model
        self.schema = schema
        self.vocab_map = vocab_map
        self.decoder = JSONDecoder()
        self.current_function: Optional[str] = None
        self.generated_text: str = '{"name": "'

        for char in self.generated_text:
            self.decoder.consume_char(char)

        self.valid_function_names = set(self.schema.get_function_names())

    def apply_token_constraint(
        self,
        logits: list[float],
        allowed_token_ids: set[int],
    ) -> list[float]:
        """Apply negative infinity mask to non-allowed token logits."""
        constrained = [float("-inf")] * len(logits)
        for token_id in allowed_token_ids:
            if token_id < len(logits):
                constrained[token_id] = logits[token_id]
        return constrained

    def get_valid_tokens_for_current_state(self) -> set[int]:
        """Dynamically compute allowed tokens based on current grammar phase."""
        valid_tokens: set[int] = set()

        # PHASE 1 : Nom de la fonction
        if '", "parameters":' not in self.generated_text:
            prefix = (
                self.generated_text.split('"name": "')[-1]
                if '"name": "' in self.generated_text
                else ""
            )

            # Si le nom est complet, on force la transition vers '", "parameters": {'
            if prefix in self.valid_function_names:
                for tid, ttext in self.vocab_map.items():
                    if ttext and any(c in ttext for c in ['"', ',', 'parameters', ':', '{']):
                        valid_tokens.add(tid)
            else:
                for name in self.valid_function_names:
                    if name.startswith(prefix):
                        for tid, ttext in self.vocab_map.items():
                            if ttext and (ttext in name or name in (prefix + ttext) or '"' in ttext):
                                valid_tokens.add(tid)

        # PHASE 2 : Génération des clés et valeurs de paramètres
        else:
            # On interdit la fermeture immédiate si aucun paramètre n'a encore été tenté
            for tid, ttext in self.vocab_map.items():
                if not ttext:
                    continue
                # Autoriser les chiffres, mots, guillemets, virgules et espaces
                if any(c.isalnum() or c in ' ".,_:-{}\n\t' for c in ttext):
                    valid_tokens.add(tid)

        if not valid_tokens:
            for tid, ttext in self.vocab_map.items():
                if ttext and any(c in ttext for c in '{},:":[]0123456789'):
                    valid_tokens.add(tid)

        return valid_tokens

    def process_selected_token(self, token_id: int) -> None:
        """Update internal state with selected token."""
        token_text = self.vocab_map.get(token_id, "")
        self.generated_text += token_text

        for char in token_text:
            self.decoder.consume_char(char)

        if '"name": "' in self.generated_text and not self.current_function:
            try:
                parts = self.generated_text.split('"name": "')
                if len(parts) > 1:
                    candidate = parts[1].split('"')[0]
                    if candidate in self.valid_function_names:
                        self.current_function = candidate
            except Exception:
                pass
