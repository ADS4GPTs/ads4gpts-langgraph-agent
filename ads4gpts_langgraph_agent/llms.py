import os
import logging
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  # Example for Anthropic
from langchain_groq import ChatGroq  # Example for Groq
from ads4gpts_langchain.toolkit import Ads4gptsToolkit
from ads4gpts_langchain.utils import get_from_dict_or_env

from ads4gpts_langgraph_agent.prompts import (
    ads4gpts_advertiser_prompt,
    ads4gpts_render_prompt,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream handler for logging
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Pre-configured models and their parameters for each provider
MODEL_CONFIG = {
    "openai": {
        "advertiser": {"model": "gpt-4o", "temperature": 0.7},
        "render": {"model": "gpt-4o", "temperature": 0.7},
    },
    "anthropic": {
        "advertiser": {"model": "claude-v1", "temperature": 0.7},
        "render": {"model": "claude-v1-mini", "temperature": 0.7},
    },
    "groq": {
        "advertiser": {"model": "groq-adv", "temperature": 0.7},
        "render": {"model": "groq-rend", "temperature": 0.7},
    },
}


def create_llm(provider: str, model_type: str, api_key: str):
    config = MODEL_CONFIG[provider][model_type]
    model = config["model"]
    temperature = config["temperature"]

    logger.info(
        f"Creating LLM for provider: {provider}, model: {model}, temperature: {temperature}"
    )

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model, temperature=temperature, anthropic_api_key=api_key
        )
    elif provider == "groq":
        return ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    else:
        logger.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")


def create_advertiser_agent(provider: str, api_key: str, ads4gpts_api_key: str):
    logger.info(f"Creating advertiser agent for provider: {provider}")
    advertiser_llm = create_llm(provider, "advertiser", api_key)
    toolkit = Ads4gptsToolkit(ads4gpts_api_key=ads4gpts_api_key).get_tools()
    return ads4gpts_advertiser_prompt | advertiser_llm.bind_tools(toolkit)


def create_render_agent(provider: str, api_key: str):
    logger.info(f"Creating render agent for provider: {provider}")
    render_llm = create_llm(provider, "render", api_key)
    return ads4gpts_render_prompt | render_llm


# Example usage
if __name__ == "__main__":
    provider = get_from_dict_or_env({}, "PROVIDER", "PROVIDER")
    api_key = get_from_dict_or_env(
        {}, f"{provider.upper()}_API_KEY", f"{provider.upper()}_API_KEY"
    )
    ads4gpts_api_key = get_from_dict_or_env({}, "ADS4GPTS_API_KEY", "ADS4GPTS_API_KEY")

    logger.info(f"Starting agents creation with provider: {provider}")
    advertiser_agent = create_advertiser_agent(provider, api_key, ads4gpts_api_key)
    render_agent = create_render_agent(provider, api_key)
    logger.info("Agents created successfully")
