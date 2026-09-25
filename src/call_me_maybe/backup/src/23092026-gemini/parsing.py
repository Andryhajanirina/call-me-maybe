"""Dynamic pipeline for prompt processing and constrained JSON output generation."""

import json
from typing import Any
from llm_sdk import Small_LLM_Model

from .function_schema import FunctionSchema
from .models import FunctionCallResult
from .token_constraints import TokenConstraints


def load_vocabulary(model: Small_LLM_Model) -> dict[int, str]:
    """Load vocabulary mapping from token IDs to string representations."""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    vocab_map: dict[int, str] = {}
    for token_str, token_id in vocab_json.items():
        vocab_map[int(token_id)] = token_str
    return vocab_map


class GenerationPipeline:
    """Manages token generation using strict dynamic constrained decoding."""

    def __init__(
        self,
        model: Small_LLM_Model,
        schema: FunctionSchema,
        max_tokens: int = 120,
    ) -> None:
        """Initialize pipeline with loaded function schema."""
        self.model = model
        self.schema = schema
        self.max_tokens = max_tokens
        self.vocab_map = load_vocabulary(model)

    def _build_prompt(self, user_prompt: str) -> str:
        """Construct system prompt dynamically from functions_definition.json schema."""
        functions_desc = []
        for fn in self.schema.functions:
            params_list = [f'"{k}": ({v.type})' for k, v in fn.parameters.items()]
            params_str = ", ".join(params_list)
            functions_desc.append(
                f"Function: {fn.name}\n"
                f"Description: {fn.description}\n"
                f"Parameters: {{{params_str}}}"
            )

        tools_str = "\n\n".join(functions_desc)

        return (
            f"Available Functions:\n{tools_str}\n\n"
            f"User Prompt: {user_prompt}\n\n"
            f"Extract arguments and respond strictly with JSON function call:\n"
            f'{{"name": "'
        )

    def process_prompt(self, prompt: str) -> FunctionCallResult:
        """Translate prompt into function call strictly via LLM logits and constraints."""
        constraints = TokenConstraints(self.model, self.schema, self.vocab_map)
        formatted_prompt = self._build_prompt(prompt)

        input_ids = self.model.encode(formatted_prompt).squeeze(0).tolist()
        generated_tokens: list[int] = []

        for _ in range(self.max_tokens):
            current_context = input_ids + generated_tokens
            logits = self.model.get_logits_from_input_ids(current_context)

            valid_tokens = constraints.get_valid_tokens_for_current_state()
            if not valid_tokens:
                break

            constrained_logits = constraints.apply_token_constraint(
                logits, valid_tokens
            )

            best_token_id = max(
                range(len(constrained_logits)),
                key=lambda idx: constrained_logits[idx],
            )

            generated_tokens.append(best_token_id)
            constraints.process_selected_token(best_token_id)

            # Détecter la fin d'un objet JSON valide contenant "parameters"
            text = constraints.generated_text
            if (
                '"parameters":' in text
                and text.count("{") == text.count("}")
                and text.rstrip().endswith("}")
            ):
                break

        full_json = constraints.generated_text
        if not full_json.endswith("}"):
            full_json += "}"

        fn_name = constraints.current_function or ""
        fn_params: dict[str, Any] = {}

        try:
            parsed = json.loads(full_json)
            if isinstance(parsed, dict):
                fn_name = str(parsed.get("name", fn_name))
                raw_params = parsed.get("parameters", {})
                if isinstance(raw_params, dict):
                    fn_params = raw_params
        except Exception:
            pass

        if not isinstance(fn_params, dict):
            fn_params = {}

        return FunctionCallResult(
            prompt=prompt,
            name=fn_name,
            parameters=fn_params,
        )
