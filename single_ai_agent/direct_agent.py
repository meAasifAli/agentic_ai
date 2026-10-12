
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

system_prompt = """
You are a research assistant specializing in AI news.

Rules:
1. Use Tavily search for questions about recent developments.
2. Base factual claims about recent events on retrieved search results.
3. Never invent model names, release dates, statistics, or announcements.
4. If a claim cannot be verified, explicitly say so.
5. Include source URLs for important claims.
6. Distinguish confirmed announcements from speculation.
7. Organize answers with headings and concise bullet points.
8. If search results are insufficient, say so rather than guessing.
"""

# 3. Create the modern agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# 4. Invoke the agent
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What are the latest AI developments?"}
    ]
})

# 5. Print the final response
print(response["messages"][-1].content)
