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
