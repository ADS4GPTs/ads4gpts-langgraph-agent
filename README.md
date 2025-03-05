# ads4gpts-langgraph-agent

ADS4GPTs LangGraph agent. Integrate Ads into AI Agents and monetize.

## Overview

The `ads4gpts-langgraph-agent` package provides a LangGraph agent that integrates advertising capabilities into AI agents. This allows for the seamless inclusion of promotional content within AI-driven conversations, enabling monetization opportunities.

The package is a specific implementation for Inline Sponsored Response Ads, which is one format that ADS4GPTs offer.

If you find this cool please give us a star ⭐️

[![GitHub Stars](https://img.shields.io/github/stars/ADS4GPTs/ads4gpts-langgraph-agent?style=social)](https://github.com/ADS4GPTs/ads4gpts-langgraph-agent/stargazers)

Star ⭐️ the general ADS4GPTs tool monorepo 

[![GitHub Stars](https://img.shields.io/github/stars/ADS4GPTs/ads4gpts?style=social)](https://github.com/ADS4GPTs/ads4gpts/stargazers)

Star ⭐️ the general ADS4GPTs open standards

[![GitHub Stars](https://img.shields.io/github/stars/ADS4GPTs/ads4gpts-openstandards?style=social)](https://github.com/ADS4GPTs/ads4gpts-openstandards/stargazers)

## Features

- **Contextual Ad Integration**: Ads are integrated into conversations based on the context, ensuring relevance and engagement.
- **Customizable UX**: Control where and how the ads are integrated within the graph.
- **Personalization & Context Blending**: Make the ads desirable and relevant to increase User retention.
- **Privacy First**: All sensitive data are being processed within your graph.
- **Support for Multiple LLM Providers**: Compatible with OpenAI, Anthropic, and Groq models.
- **Logging**: Configurable logging for monitoring and debugging.

## Installation

To install the package, use pip or any other package manager:

```sh
pip install ads4gpts-langgraph-agent
```

## Usage

The `ads4gpts-langgraph-agent` is meant to be part of a single or multi-agent architectures, where the application aims to monetize with advertising. Key differentiator from all other solutions is that the Ad integration becomes part of the AI application UX. 

You can control the Ad placement deterministically by introducing the `ads4gpts-langgraph-agent` as a node in the LangGraph graph:

```py
from langgraph.graph import StateGraph, START, END
# State needs to have a messages field
# ConfigSchema needs to have gpt_id and session_id fields
from your_repo import ConfigSchema, State 
from ads4gpts_langgraph_agent import make_ads4gpts_langgraph_agent

graph_builder = StateGraph(State, ConfigSchema)
graph_builder.add_node("ads4gpts_node", make_ads4gpts_langgraph_agent())
```

Follow the subgraph_example in the examples folder for a complete implementation of this approach.

On the other hand you let other agents call the `ads4gpts-langgraph-agent` autonomously through tool execution:

```py
from langgraph.graph import StateGraph, START, END
# State needs to have a messages field
# ConfigSchema needs to have gpt_id and session_id fields
from your_repo import ConfigSchema, State 
from ads4gpts_langgraph_agent import make_ads4gpts_langgraph_agent, make_handoff_tool

ads4gpts_agent =  make_ads4gpts_langgraph_agent()
ads4gpts_tool = make_handoff_tool(ads4gpts_agent)
```

Follow the tool_example in the examples folder for a complete implementation of the tool approach.

## Configuration

The agent requires several configuration parameters, which can be provided through environment variables or directly in the code through the `make_ads4gpts_langgraph_agent` constructor function:

- ADS4GPTS_API_KEY: API key for ADS4GPTs.
- ADS4GPTS_BASE_URL: Base URL for ADS4GPTs API. You can omit that to use the default settings.
- ADS4GPTS_ADS_ENDPOINT: Endpoint for fetching ads. You can omit that to use the default settings.
- PROVIDER: LLM provider (e.g., openai, anthropic, groq).
- {PROVIDER}_API_KEY: API key for the selected LLM provider e.g. OPENAI_API_KEY.

## Testing
To run the tests, use the following command:

```sh
pytest tests/
```

## License
This project is licensed under the GNU AGPLv3 License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact
For any questions or inquiries, please contact the ADS4GPTs team at contact@ads4gpts.com