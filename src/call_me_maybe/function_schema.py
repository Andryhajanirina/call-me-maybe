#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   function_schema.py                                   :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:41:52 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/25 12:46:03 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from pathlib import Path

from .models import FunctionDefinition


class FunctionSchema:
    # def __init__(self, path: str | Path) -> None:
    #     self.path = Path(path)

    #     with self.path.open("r", encoding="utf-8") as file:
    #         raw_data = json.load(file)

    #     self.functions = [
    #         FunctionDefinition.model_validate(item)
    #         for item in raw_data
    #     ]
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
