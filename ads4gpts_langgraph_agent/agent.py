from typing_extensions import Literal, Annotated
from typing import Dict, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langchain_core.runnables.config import RunnableConfig

from ads4gpts_langchain.utils import get_from_dict_or_env
from ads4gpts_langchain import Ads4gptsToolkit

from ads4gpts_langgraph_agent.state import ADS4GPTsState, ADS4GPTsConfig
from ads4gpts_langgraph_agent.tools import make_handoff_tool
from ads4gpts_langgraph_agent.llms import create_advertiser_agent, create_render_agent


def make_ads4gpts_langgraph_agent(
    # agent_name: Optional[str],
    **kwargs,
):
    ads4gpts_api_key = get_from_dict_or_env(
        kwargs, "ADS4GPTS_API_KEY", "ADS4GPTS_API_KEY"
    )
    ads4gpts_toolkit = Ads4gptsToolkit(
        base_url="https://ads-api-dev.onrender.com/",
        ads_endpoint="api/v1/ads",
        ads4gpts_api_key=ads4gpts_api_key,
    ).get_tools()
    provider = get_from_dict_or_env(kwargs, "PROVIDER", "PROVIDER")
    api_key = get_from_dict_or_env(
        kwargs,
        f"{provider.upper()}_API_KEY",
        f"{provider
        .upper()}_API_KEY",
    )
    advertiser_agent = create_advertiser_agent(provider, api_key, ads4gpts_api_key)
    render_agent = create_render_agent(provider, api_key)

    async def advertiser_node(state: ADS4GPTsState, config: RunnableConfig):
        session_id = config["configurable"].get("session_id", "")
        gpt_id = config["configurable"].get("gpt_id", "")
        advertiser_agent_response = await advertiser_agent.ainvoke(
            {
                "messages": state["messages"],
                "ad_prompt": "Get one Inline Ad based on the context provided.",
                "session_id": session_id,
                "gpt_id": gpt_id,
            }
        )
        return {"messages": [advertiser_agent_response]}

    async def render_node(state: ADS4GPTsState, config: RunnableConfig):
        render_agent_response = await render_agent.ainvoke(
            {"ads": state["messages"][-1].content, "messages": state["messages"][:-2]}
        )
        return {"messages": [render_agent_response]}

    ads4gpts_tool_node = ToolNode(ads4gpts_toolkit)

    graph = StateGraph(ADS4GPTsState, ADS4GPTsConfig)
    graph.add_node("advertiser_node", advertiser_node)
    graph.add_node("ads4gpts_tool_node", ads4gpts_tool_node)
    graph.add_node("render_node", render_node)
    graph.add_edge(START, "advertiser_node")
    graph.add_edge("advertiser_node", "ads4gpts_tool_node")
    graph.add_edge("ads4gpts_tool_node", "render_node")

    return graph.compile()
