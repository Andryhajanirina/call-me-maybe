import json
from pathlib import Path

from .models import FunctionDefinition


class FunctionSchema:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        with self.path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)

        self.functions = [
            FunctionDefinition.model_validate(item)
            for item in raw_data
        ]

    def get_function(self, name: str) -> FunctionDefinition | None:
        for function in self.functions:
            if function.name == name:
                return function

        return None

    def get_function_names(self) -> list[str]:
        return [function.name for function in self.functions]
