import argparse
# Mbola tsy miasa


def parse_args() -> argparse.Namespace:
    """
    Parse the uv run arguments:
    --functions_definition with
        default data/input/functions_definition.json
    --input with default data/input/function_calling_tests.json
    --output with default data/output/function_calling_results.json
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        help="Path to the JSON file containing "
                        "function definitions")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="Path to the JSON file containing input prompts")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json",
                        help="Path where generated function calls "
                        "will be saved")
    parser.add_argument("--model",
                        default="Qwen/Qwen3-0.6B",
                        help="Name of the language model to use")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print detailed generation trace to stdout")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(args._get_kwargs()[0][1])
