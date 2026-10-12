
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()

# 1. Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# 2. Initialize Tavily search
search_tool = TavilySearch(max_results=2)
tools = [search_tool]

# 3. System prompt
system_prompt = """
You are a helpful AI research assistant.

- Use Tavily to search for current information when needed.
- Base factual claims on the search results.
- Do not invent facts, dates, statistics, or source URLs.
- Clearly explain when information cannot be verified.
- Provide concise, well-structured answers.
"""

# 4. Create the modern agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

# 5. Stream the agent's response
inputs = {
    "messages": [
        {
            "role": "user",
            "content": "What are the latest AI developments?",
        }
    ]
}

for token, metadata in agent.stream(
    inputs,
    stream_mode="messages",
):
    if token.content:
        if isinstance(token.content, str):
            print(token.content, end="", flush=True)
        elif isinstance(token.content, list):
            for block in token.content:
                if block.get("type") == "text":
                    print(block["text"], end="", flush=True)

print()
