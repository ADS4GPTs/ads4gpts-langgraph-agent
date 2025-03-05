import os
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

chat_system_prompt = "You are a helpful assistant"
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chat_system_prompt),
        MessagesPlaceholder("messages", optional=True),
    ]
)
chat_llm = ChatOpenAI(
    model="gpt-4o", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY")
)
chat_agent = chat_prompt | chat_llm


from pydantic import BaseModel, Field


supervisor_system_prompt = """
# Role
You are a supervisor of AI agents. Your role is select the best agent for the task.

# Objective
Your objective is to route to the best agent to maximize the user experience and your own profit.

#Instructions
1. All conversational responses should be routed to chat agent.
2. After each chat AI response you decide if the user would benefit from a relevant advertisement and call the appropriate agent. 
3. End the conversation round after an Ad or after a chat if an Ad is not relevant.
"""
supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_system_prompt),
        MessagesPlaceholder("messages"),
    ]
)
supervisor_llm = ChatOpenAI(
    model="gpt-4o", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY")
)
