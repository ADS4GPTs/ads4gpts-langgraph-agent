from typing import Annotated

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import ToolMessage


def make_handoff_tool(*, agent_name: str, parent: bool = False):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig,
    ):
        """Ask another agent for help."""
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            tool_call_id=tool_call_id,
        )
        if parent:
            return Command(
                goto=agent_name,
                graph=Command.PARENT,
                update={"messages": [tool_message]},
            )
        else:
            return Command(
                goto=agent_name,
                update={"messages": [tool_message]},
            )

    return handoff_to_agent
