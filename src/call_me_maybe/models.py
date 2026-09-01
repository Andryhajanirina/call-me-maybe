from typing import Literal

from pydantic import BaseModel


ParameterType = Literal["string", "number"]


class ParameterDefinition(BaseModel):
    type: ParameterType


class ReturnDefinition(BaseModel):
    type: ParameterType


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition
