import operator
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage


class ADS4GPTsState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class ADS4GPTsConfig:
    session_id: str
    gpt_id: str
