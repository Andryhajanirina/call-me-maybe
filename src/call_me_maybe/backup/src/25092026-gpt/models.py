#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   models.py                                            :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:42:22 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/22 10:07:23 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from typing import Any, Literal

from pydantic import BaseModel, Field


# ParameterType = Literal["string", "number"]
ParameterType = Literal["string", "number", "boolean", "array", "object"]


class ParameterDefinition(BaseModel):
    """Represents a function parameter definition in JSON schema.

    Attributes:
        type: The data type of the parameter.
        description: Optional description of the parameter.
    """
    type: ParameterType
    description: str = ""


class ReturnDefinition(BaseModel):
    """Represents the return type definition of a function.

    Attributes:
        type: The data type returned by the function.
    """
    type: ParameterType


class FunctionDefinition(BaseModel):
    """Represents a callable function definition.

    Attributes:
        name: Name of the function.
        description: Description of what the function does.
        parameters: Dictionary mapping argument names to their
            type definitions.
        returns: Return type definition of the function.
    """
    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ReturnDefinition


class TestInput(BaseModel):
    """Represents an input prompt test case.

    Attributes:
        prompt: Natural language request to translate into a function call.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """Represents the structured JSON output for a function call.

    Attributes:
        prompt: Original natural language prompt.
        name: Name of the selected function.
        parameters: Extracted typed arguments for the function.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
