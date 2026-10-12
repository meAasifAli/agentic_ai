
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()

st.set_page_config(
    page_title="Direct AI Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Direct AI Research Agent")
st.caption("Powered by LangChain, Groq, and Tavily")


@st.cache_resource
def create_research_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    search_tool = TavilySearch(
        max_results=3,
    )

    # Give the tool a clear description
    search_tool.description = (
        "Search the web for current events, news, and factual information. "
        "You MUST provide a non-empty string in the required 'query' field. "
        "Example: {'query': 'Latest US Iran relations news'}."
    )

    system_prompt = """
    You are a helpful AI research assistant.

    TOOL USAGE:
    - Use tavily_search for current events and recent information.
    - When calling tavily_search, ALWAYS provide the required argument:
      query: a non-empty string containing the search terms.
    - Never call the search tool with missing or empty arguments.
    - If a search fails, do not invent search results.
    - Base factual claims on retrieved results.
    - Do not fabricate dates, statistics, citations, or URLs.
    - Give concise, well-structured answers.
    - For current affairs, explain when information could not be verified.
    """

    return create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=system_prompt,
    )


agent = create_research_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            inputs = {
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            }

            for token, metadata in agent.stream(
                inputs,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") != "model":
                    continue

                # Ignore tool-call chunks
                if token.tool_calls:
                    continue

                content = token.content

                if isinstance(content, str):
                    full_response += content

                elif isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            full_response += block.get("text", "")

                if full_response:
                    response_placeholder.markdown(
                        full_response + "▌"
                    )

            response_placeholder.markdown(
                full_response or "No response was generated."
            )

        except Exception as exc:
            st.error(f"An error occurred: {exc}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )
