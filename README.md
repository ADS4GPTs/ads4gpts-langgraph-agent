# ads4gpts-langgraph-agent

ADS4GPTs LangGraph agent. Integrate Ads into AI Agents and monetize.

## Overview

The `ads4gpts-langgraph-agent` package provides a LangGraph agent that integrates advertising capabilities into AI agents. This allows for the seamless inclusion of promotional content within AI-driven conversations, enabling monetization opportunities.

## Features

- **Contextual Ad Integration**: Ads are integrated into conversations based on the context, ensuring relevance and engagement.
- **Customizable UX**: Control where and how the ads are integrated within the graph.
- **Personalization & Context Blending**: Make the ads desirable and relevant to increase User retention.
- **Privacy First**: All sensitive data are being processed within your graph.
- **Support for Multiple LLM Providers**: Compatible with OpenAI, Anthropic, and Groq models.
- **Logging**: Configurable logging for monitoring and debugging.

## Installation

To install the package, use the following command:

```sh
pip install ads4gpts-langgraph-agent
```

## Usage

The `ads4gpts-langgraph-agent` package is meant to be used as a node within a graph.

```py
from langgraph.graph import StateGraph, START, END
# State needs to have a messages field
# ConfigSchema needs to have gpt_id and session_id fields
from your_repo import ConfigSchema, State 
from ads4gpts_langgraph_agent import make_ads4gpts_langgraph_agent

graph_builder = StateGraph(State, ConfigSchema)
graph_builder.add_node("ads4gpts_node", make_ads4gpts_langgraph_agent())
```


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