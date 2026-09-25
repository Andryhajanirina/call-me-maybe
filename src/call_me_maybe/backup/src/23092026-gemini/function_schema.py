#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   function_schema.py                                   :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:41:52 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/09 11:42:01 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

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
