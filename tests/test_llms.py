import pytest
from unittest.mock import MagicMock
from ads4gpts_langgraph_agent.llms import (
    create_llm,
    create_advertiser_agent,
    create_render_agent,
)


def test_create_llm_openai(mocker):
    api_key = "test_openai_api_key"
    provider = "openai"
    model_type = "advertiser"

    mock_openai = mocker.patch("ads4gpts_langgraph_agent.llms.ChatOpenAI")

    llm = create_llm(provider, model_type, api_key)

    mock_openai.assert_called_once_with(
        model="gpt-4o", temperature=0.7, openai_api_key=api_key
    )
    assert llm == mock_openai.return_value


def test_create_llm_anthropic(mocker):
    api_key = "test_anthropic_api_key"
    provider = "anthropic"
    model_type = "advertiser"

    mock_anthropic = mocker.patch("ads4gpts_langgraph_agent.llms.ChatAnthropic")

    llm = create_llm(provider, model_type, api_key)

    mock_anthropic.assert_called_once_with(
        model="claude-v1", temperature=0.7, anthropic_api_key=api_key
    )
    assert llm == mock_anthropic.return_value


def test_create_llm_groq(mocker):
    api_key = "test_groq_api_key"
    provider = "groq"
    model_type = "advertiser"

    mock_groq = mocker.patch("ads4gpts_langgraph_agent.llms.ChatGroq")

    llm = create_llm(provider, model_type, api_key)

    mock_groq.assert_called_once_with(
        model="groq-adv", temperature=0.7, groq_api_key=api_key
    )
    assert llm == mock_groq.return_value


def test_create_advertiser_agent(mocker):
    provider = "openai"
    api_key = "test_openai_api_key"
    ads4gpts_api_key = "test_ads4gpts_api_key"

    mock_create_llm = mocker.patch("ads4gpts_langgraph_agent.llms.create_llm")
    mock_toolkit = mocker.patch("ads4gpts_langgraph_agent.llms.Ads4gptsToolkit")

    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm
    mock_toolkit_instance = MagicMock()
    mock_toolkit.return_value.get_tools.return_value = mock_toolkit_instance

    agent = create_advertiser_agent(provider, api_key, ads4gpts_api_key)

    mock_create_llm.assert_called_once_with(provider, "advertiser", api_key)
    mock_toolkit.assert_called_once_with(ads4gpts_api_key=ads4gpts_api_key)
    mock_llm.bind_tools.assert_called_once_with(mock_toolkit_instance)
    assert agent is not None


def test_create_render_agent(mocker):
    provider = "openai"
    api_key = "test_openai_api_key"

    mock_create_llm = mocker.patch("ads4gpts_langgraph_agent.llms.create_llm")

    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm

    agent = create_render_agent(provider, api_key)

    mock_create_llm.assert_called_once_with(provider, "render", api_key)
    assert agent is not None
