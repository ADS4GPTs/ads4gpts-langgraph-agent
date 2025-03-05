import os
import logging
import operator
from typing import Optional, List
from typing_extensions import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

memory = MemorySaver()

from ads4gpts_langgraph_agent import make_ads4gpts_langgraph_agent
from ads4gpts_langgraph_agent.tools import make_handoff_tool

from ads4gpts_langchain.utils import get_from_dict_or_env

from llms import (
    supervisor_llm,
    supervisor_prompt,
    chat_prompt,
    chat_llm,
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


ads4gpts_agent = make_ads4gpts_langgraph_agent()
ads4gpts_tool = make_handoff_tool(agent_name="ads4gpts_agent")

chat_model = chat_prompt | chat_llm


async def chat_agent_node(state: State, config: RunnableConfig):
    """A simple chat node that returns a static response."""
    logger.info("Chat agent node invoked.")
    chat_response = await chat_model.ainvoke(
        {
            "messages": state["messages"],
        }
    )
    return {"messages": [chat_response]}


chat_builder = StateGraph(State, ConfigSchema)
chat_builder.add_node("chat_agent_node", chat_agent_node)
chat_builder.add_edge(START, "chat_agent_node")
chat_builder.add_edge("chat_agent_node", END)
chat_agent = chat_builder.compile()
chat_tool = make_handoff_tool(agent_name="chat_agent")

tools = [ads4gpts_tool, chat_tool]
tool_node = ToolNode(tools)

supervisor_agent = supervisor_prompt | supervisor_llm.bind_tools(tools)


async def supervisor_node(state: State, config: RunnableConfig):
    """A simple supervisor node that returns a decision."""
    logger.info("Supervisor node invoked.")
    supervisor_response = await supervisor_agent.ainvoke(
        {
            "messages": state["messages"],
        }
    )
    return {"messages": [supervisor_response]}


def supervisor_edge(state: State, config: RunnableConfig) -> Literal["tool_node", END]:
    """A simple edge function that returns a decision."""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    else:
        return END


# async def supervisor_node(
#     state: State, config: RunnableConfig
# ) -> Command[Literal["tool_node", END]]:
#     """A simple supervisor node that returns a decision."""
#     logger.info("Supervisor node invoked.")
#     supervisor_response = await supervisor_agent.ainvoke(
#         {
#             "messages": state["messages"],
#         }
#     )
#     if supervisor_response.tool_calls:
#         return Command(
#             goto="tool_node",
#             update={"messages": [supervisor_response]},
#             tool_calls=supervisor_response.tool_calls,
#         )
#     else:
#         return Command(
#             goto=END,
#             update={"messages": [supervisor_response]},
#         )


graph_builder = StateGraph(State, ConfigSchema)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("chat_agent", chat_agent)
graph_builder.add_node("ads4gpts_agent", ads4gpts_agent)
graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", supervisor_edge)
graph = graph_builder.compile()
