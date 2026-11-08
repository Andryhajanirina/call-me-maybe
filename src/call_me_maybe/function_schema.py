#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   function_schema.py                                   :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:41:52 by andry-ha            #+#    #+#            #
#   Updated: 2026/11/08 13:58:19 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from pathlib import Path
from typing import Any

from .models import FunctionDefinition


class FunctionSchema:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        try:
            with self.path.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)
                self.functions = [
                    FunctionDefinition.model_validate(item)
                    for item in raw_data
                ]
        except FileNotFoundError as e:
            raise ValueError(
                f"Function definition file not found: {self.path}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in function definition file: {self.path}"
            ) from e

    def get_function(self, name: str) -> FunctionDefinition | None:
        for function in self.functions:
            if function.name == name:
                return function

        return None

    def normalize_parameters(
        self,
        function_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize parameter values according to their schema types."""
        function = self.get_function(function_name)

        if function is None:
            raise ValueError(
                f"Unknown function: {function_name}"
            )

        normalized: dict[str, Any] = {}

        for name, value in parameters.items():
            parameter = function.parameters.get(name)

            if parameter is None:
                raise ValueError(
                    f"Unknown parameter '{name}' "
                    f"for function '{function_name}'"
                )

            parameter_type = parameter.type

            if parameter_type == "number":
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ValueError(
                        f"Parameter '{name}' must be a number"
                    )

                normalized[name] = float(value)

            elif parameter_type in ("int", "integer"):
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ValueError(
                        f"Parameter '{name}' must be an integer"
                    )

                if isinstance(value, float) and not value.is_integer():
                    raise ValueError(
                        f"Parameter '{name}' must be an integer, "
                        f"got {value}"
                    )

                normalized[name] = int(value)

            elif parameter_type == "string":
                if not isinstance(value, str):
                    raise ValueError(
                        f"Parameter '{name}' must be a string"
                    )

                normalized[name] = value

            elif parameter_type == "boolean":
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Parameter '{name}' must be a boolean"
                    )

                normalized[name] = value

            elif parameter_type == "array":
                if not isinstance(value, list):
                    raise ValueError(
                        f"Parameter '{name}' must be an array"
                    )

                normalized[name] = value

            elif parameter_type == "object":
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Parameter '{name}' must be an object"
                    )

                normalized[name] = value

        missing_parameters = (
            set(function.parameters) - set(parameters)
        )

        if missing_parameters:
            missing = ", ".join(sorted(missing_parameters))
            raise ValueError(
                f"Missing parameter(s) for function "
                f"'{function_name}': {missing}"
            )

        return normalized

    def get_function_names(self) -> list[str]:
        return [function.name for function in self.functions]

    def get_functions_description(self) -> str:
        """Build a description of all available functions."""
        descriptions = []

        for function in self.functions:
            parameters = ", ".join(
                f"{name}: {parameter.type}"
                for name, parameter in function.parameters.items()
            )

            descriptions.append(
                f"{function.name}({parameters}): "
                f"{function.description}"
            )

        return "\n".join(descriptions)
