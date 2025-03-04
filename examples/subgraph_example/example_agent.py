import os
import logging
import operator
from typing import Optional, List
from typing_extensions import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from ads4gpts_langgraph_agent import make_ads4gpts_langgraph_agent

from ads4gpts_langchain.utils import get_from_dict_or_env

from llms import (
    supervisor_agent,
    chat_agent,
    SupervisorDecision,
    SupervisorDecisionEnum,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream handler for logging
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


class ConfigSchema(TypedDict):
    session_id: str
    gpt_id: str
    user_id: str


async def supervisor_node(
    state: State, config: RunnableConfig
) -> Command[Literal["chat_agent_node", "ads4gpts_node", END]]:
    """A simple supervisor node that returns a decision."""
    logger.info("Supervisor node invoked.")
    supervisor_response = await supervisor_agent.ainvoke(
        {
            "messages": state["messages"],
        }
    )
    return Command(
        goto=supervisor_response.decision.value,
        update={"messages": [AIMessage("Routing to " + supervisor_response.decision)]},
    )


async def chat_agent_node(state: State, config: RunnableConfig):
    """A simple chat node that returns a message."""
    logger.info("Chat node invoked.")
    chat_agent_response = await chat_agent.ainvoke(
        {
            "messages": state["messages"],
        }
    )
    return {"messages": [chat_agent_response]}


# def supervisor_edge(
#     state: State, config: RunnableConfig
# ) -> Literal["chat_agent_node", "ads4gpts_node", "__end__"]:
#     """A simple edge function that returns a decision."""
#     logger.info("Supervisor edge invoked.")
#     last_message = state["messages"][-1]
#     print(last_message)
#     if last_message.decision == SupervisorDecisionEnum.CHAT:
#         return "chat_agent_node"
#     elif last_message.decision == SupervisorDecisionEnum.ADS:
#         return "ads4gpts_node"
#     else:
#         return "__end__"


graph_builder = StateGraph(State, ConfigSchema)
graph_builder.add_node("supervisor_node", supervisor_node)
graph_builder.add_node("chat_agent_node", chat_agent_node)
graph_builder.add_node("ads4gpts_node", make_ads4gpts_langgraph_agent())
graph_builder.add_edge(START, "supervisor_node")
graph_builder.add_edge("chat_agent_node", "supervisor_node")
graph_builder.add_edge("ads4gpts_node", "supervisor_node")
graph = graph_builder.compile(checkpointer=memory)
