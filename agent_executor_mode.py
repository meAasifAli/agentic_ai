from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
)

search_tool = TavilySearch(max_results=2)
tools = [search_tool]

prompt = PromptTemplate.from_template("""
Answer the following question using the available tools.

{tools}

Use this format:
Question: {input}
Thought: your reasoning
Action: one of [{tool_names}]
Action Input: the input to the tool
Observation: the tool result
... repeat as needed
Thought: I now know the answer
Final Answer: your answer

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({
    "input": "What are the latest AI developments?"
})

print(response["output"])