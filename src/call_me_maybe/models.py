#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   models.py                                            :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:42:22 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/09 11:42:36 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from typing import Literal

from pydantic import BaseModel


ParameterType = Literal["string", "number"]


class ParameterDefinition(BaseModel):
    """
    Represents the type for each parameter on the
    functions_definition.json - JSON file:
    "string", "number", "integer", and "boolean"
    """
    type: ParameterType


class ReturnDefinition(BaseModel):
    type: ParameterType


class FunctionDefinition(BaseModel):
    """
    Represents each field on JSON:
    functions_definition.json
    name,description,parameters,returns
    """
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition
