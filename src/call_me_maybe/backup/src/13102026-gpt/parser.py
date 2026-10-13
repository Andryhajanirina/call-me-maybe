import json
from typing import Any, Optional


def load_data(file: str) -> list[dict]:
    try:
        with open(file) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file '{file}'")
    except PermissionError:
        raise ValueError("Permission denied")
    except FileNotFoundError:
        raise ValueError(f"File not found '{file}'")
    return data


def main():
    print("=== TESTING PROMPT ===")
    prompts = load_data("../../data/input/function_calling_tests.json")
    fn_names = load_data("../../data/input/functions_definition.json")

    for p, fn in zip(prompts, fn_names):
        print(f"prompt: {p['prompt']}")
        print(f"name: {fn['name']}")


if __name__ == "__main__":
    main()
