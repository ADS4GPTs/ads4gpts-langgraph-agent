from unittest.mock import MagicMock, patch
from ads4gpts_langgraph_agent.agent import make_ads4gpts_langgraph_agent
from langgraph.graph import StateGraph
import pytest


@patch("ads4gpts_langgraph_agent.agent.Ads4gptsToolkit")
@patch("ads4gpts_langgraph_agent.agent.create_advertiser_agent")
@patch("ads4gpts_langgraph_agent.agent.create_render_agent")
def test_make_ads4gpts_langgraph_agent(
    mock_create_render_agent,
    mock_create_advertiser_agent,
    mock_Ads4gptsToolkit,
):
    # Mock Ads4gptsToolkit instance and its get_tools method
    mock_toolkit_instance = MagicMock()
    mock_Ads4gptsToolkit.return_value.get_tools.return_value = mock_toolkit_instance

    # Mock advertiser and render agents
    mock_advertiser_agent = MagicMock()
    mock_create_advertiser_agent.return_value = mock_advertiser_agent
    mock_render_agent = MagicMock()
    mock_create_render_agent.return_value = mock_render_agent

    # Call the function with test arguments
    agent_name = "test_agent"
    kwargs = {
        "ADS4GPTS_API_KEY": "test_ads4gpts_api_key",
        "PROVIDER": "openai",
        "OPENAI_API_KEY": "test_openai_api_key",
    }
    graph = make_ads4gpts_langgraph_agent(agent_name, **kwargs)

    # Assertions
    mock_Ads4gptsToolkit.assert_called_once_with(
        ads4gpts_api_key="test_ads4gpts_api_key"
    )
    mock_create_advertiser_agent.assert_called_once_with(
        "openai", "test_openai_api_key", "test_ads4gpts_api_key"
    )
    mock_create_render_agent.assert_called_once_with(
        "openai", "test_openai_api_key", "test_ads4gpts_api_key"
    )

    assert graph is not None
    assert isinstance(graph, StateGraph)


@patch("ads4gpts_langgraph_agent.agent.Ads4gptsToolkit")
@patch("ads4gpts_langgraph_agent.agent.create_advertiser_agent")
@patch("ads4gpts_langgraph_agent.agent.create_render_agent")
def test_make_ads4gpts_langgraph_agent_missing_api_key(
    mock_create_render_agent,
    mock_create_advertiser_agent,
    mock_Ads4gptsToolkit,
):
    # Mock Ads4gptsToolkit instance and its get_tools method
    mock_toolkit_instance = MagicMock()
    mock_Ads4gptsToolkit.return_value.get_tools.return_value = mock_toolkit_instance

    # Mock advertiser and render agents
    mock_advertiser_agent = MagicMock()
    mock_create_advertiser_agent.return_value = mock_advertiser_agent
    mock_render_agent = MagicMock()
    mock_create_render_agent.return_value = mock_render_agent

    # Call the function with missing API key
    agent_name = "test_agent"
    kwargs = {
        "PROVIDER": "openai",
        "OPENAI_API_KEY": "test_openai_api_key",
    }

    with pytest.raises(KeyError):
        make_ads4gpts_langgraph_agent(agent_name, **kwargs)


@patch("ads4gpts_langgraph_agent.agent.Ads4gptsToolkit")
@patch("ads4gpts_langgraph_agent.agent.create_advertiser_agent")
@patch("ads4gpts_langgraph_agent.agent.create_render_agent")
def test_make_ads4gpts_langgraph_agent_invalid_provider(
    mock_create_render_agent,
    mock_create_advertiser_agent,
    mock_Ads4gptsToolkit,
):
    # Mock Ads4gptsToolkit instance and its get_tools method
    mock_toolkit_instance = MagicMock()
    mock_Ads4gptsToolkit.return_value.get_tools.return_value = mock_toolkit_instance

    # Mock advertiser and render agents
    mock_advertiser_agent = MagicMock()
    mock_create_advertiser_agent.return_value = mock_advertiser_agent
    mock_render_agent = MagicMock()
    mock_create_render_agent.return_value = mock_render_agent

    # Call the function with an invalid provider
    agent_name = "test_agent"
    kwargs = {
        "ADS4GPTS_API_KEY": "test_ads4gpts_api_key",
        "PROVIDER": "invalid_provider",
        "INVALID_PROVIDER_API_KEY": "test_invalid_provider_api_key",
    }

    with pytest.raises(KeyError):
        make_ads4gpts_langgraph_agent(agent_name, **kwargs)
    mock_get_from_dict_or_env.assert_any_call(
        kwargs, "OPENAI_API_KEY", "OPENAI_API_KEY"
    )
    mock_Ads4gptsToolkit.assert_called_once_with(
        ads4gpts_api_key="test_ads4gpts_api_key"
    )
    mock_create_advertiser_agent.assert_called_once_with(
        "openai", "test_openai_api_key", "test_ads4gpts_api_key"
    )
    mock_create_render_agent.assert_called_once_with(
        "openai", "test_openai_api_key", "test_ads4gpts_api_key"
    )

    assert graph is not None
    assert isinstance(graph, StateGraph)
