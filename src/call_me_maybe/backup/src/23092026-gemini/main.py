#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:43:04 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/22 14:02:23 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

"""Main entry point for function calling tool."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model

from .function_schema import FunctionSchema
from .models import FunctionCallResult
from .parsing import GenerationPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Translate prompts to function calls using constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    return parser.parse_args()


def load_json_file(filepath: str) -> Any:
    """Safely load and parse a JSON file with error handling.

    Args:
        filepath: Path to JSON file.

    Returns:
        Parsed JSON object.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as err:
        print(f"Error: Invalid JSON format in {filepath}: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Error reading {filepath}: {err}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Execute the function calling translation process."""
    args = parse_args()

# 1. Initialiser le schéma de fonctions avec le CHEMIN du fichier (string/Path)
    schema = FunctionSchema(args.functions_definition)

    # 2. Charger les prompts de test
    tests_data = load_json_file(args.input)

    # 3. Instancier le modèle LLM du SDK
    model = Small_LLM_Model()

    # 4. Initialiser le pipeline
    pipeline = GenerationPipeline(model, schema)

    results: list[dict[str, Any]] = []

    # 5. Traiter chaque prompt
    for item in tests_data:
        prompt = item.get("prompt", "")
        if not prompt:
            continue

        res: FunctionCallResult = pipeline.process_prompt(prompt)
        print(res.model_dump())
        results.append(res.model_dump())

    # 6. Écrire le fichier de sortie
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Successfully generated function calls in: {args.output}")
    except Exception as err:
        print(f"Error writing output file {args.output}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
