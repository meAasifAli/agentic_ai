# Single AI Agent: Notes

Study notes for the `single_ai_agent` module. Each section covers one script: what it does, how it works, and what to take away from it.

---

## Core concepts

### What is an agent?
An **agent** is an LLM running in a loop that can decide to call **tools** (functions) and then use their results to produce an answer:

```
User question
   ↓
LLM decides ──► needs info? ──► call tool ──► observation ──┐
   ▲                                                         │
   └─────────────────────────────────────────────────────────┘
   ↓ (enough info)
Final answer
```

### The building blocks used in every script
| Block | Code | Role |
|---|---|---|
| **LLM** | `ChatGroq(model=..., temperature=0)` | The model that reasons and picks tools. `temperature=0` gives deterministic, factual output. |
| **Tools** | `TavilySearch(max_results=N)` or a custom `@tool` function | Actions the model is allowed to take. |
| **Prompt** | `system_prompt` / `PromptTemplate` | Rules for behaviour: when to use tools, not making facts up, output format. |
| **Agent** | `create_agent(...)` or `create_react_agent(...)` + `AgentExecutor` | Connects the LLM, tools, and prompt into a loop. |
| **Secrets** | `load_dotenv()` | Reads `GROQ_API_KEY`, `TAVILY_API_KEY`, `WEATHERSTACK_API_KEY` from `.env`. |

### Two generations of LangChain agents
| | Legacy (`langchain_classic`) | Modern (`langchain` v1) |
|---|---|---|
| Constructor | `create_react_agent(llm, tools, prompt)` | `create_agent(model, tools, system_prompt)` |
| Runner | Wrap in `AgentExecutor(...)` | The returned agent is runnable (a LangGraph graph) |
| Tool calling | The LLM writes `Action:` / `Action Input:` text, which is parsed with regex | Native structured **tool calling** from the model API |
| Input | `{"input": "..."}` | `{"messages": [{"role": "user", "content": "..."}]}` |
| Output | `response["output"]` | `response["messages"][-1].content` |
| Streaming | Limited | `agent.stream(..., stream_mode="messages")` |

**Takeaway:** use `create_agent` for new code. The ReAct example is there to show how agents worked before native tool calling.

---

## 1. `agent_executor_mode.py`: legacy ReAct agent

**Model:** `qwen/qwen3.8-27b` on Groq · **Tool:** Tavily (2 results)

- Builds a **ReAct** prompt template with placeholders that `create_react_agent` fills in:
  - `{tools}`: tool names and descriptions
  - `{tool_names}`: the allowed values for `Action:`
  - `{input}`: the user question
  - `{agent_scratchpad}`: the running log of previous Thought/Action/Observation steps
- `AgentExecutor` runs the loop:
  - `verbose=True` prints every reasoning step to the console, which is useful for learning.
  - `handle_parsing_errors=True` sends badly formatted LLM output back to the model so it can retry, instead of crashing.
- Invoked with `{"input": "What are the latest AI developments?"}`. The answer is read from `response["output"]`.

**Takeaway:** ReAct depends on the LLM following a text format exactly. If the format slips, parsing breaks, which is why `handle_parsing_errors` exists.

---

## 2. `direct_agent.py`: modern agent with web search

**Model:** `openai/gpt-oss-120b` on Groq · **Tool:** Tavily (2 results)

- One call creates the agent: `create_agent(model=llm, tools=tools, system_prompt=system_prompt)`.
- The **system prompt** sets up a careful research assistant:
  - Search for recent developments; base claims on the results.
  - Never invent model names, dates, statistics, or announcements.
  - Cite source URLs; separate confirmed facts from speculation.
  - Use headings and bullet points; say so when the results are not enough.
- Input is a list of chat messages. The final answer is the last message: `response["messages"][-1].content`.

**Takeaway:** most of an agent's reliability comes from its system prompt. Write explicit anti-hallucination and citation rules.

---

## 3. `direct_agent_response_streaming.py`: streaming tokens

Same agent as #2 with a shorter system prompt, but it **streams** output instead of waiting for the full answer:

```python
for token, metadata in agent.stream(inputs, stream_mode="messages"):
    if token.content:
        if isinstance(token.content, str):
            print(token.content, end="", flush=True)
        elif isinstance(token.content, list):        # content blocks
            for block in token.content:
                if block.get("type") == "text":
                    print(block["text"], end="", flush=True)
```

- `stream_mode="messages"` yields `(message_chunk, metadata)` pairs as the LLM generates tokens.
- `content` can be a **string** or a **list of content blocks** (e.g. `{"type": "text", ...}`), so the code handles both.
- `end="", flush=True` prints tokens on one line, immediately.

**Takeaway:** streaming makes the agent feel faster because the user sees text as soon as it is generated.

**Caveat:** this version prints chunks from every node, so tool output may also appear in the stream. The Streamlit version (#5) filters this out.

---

## 4. `weather_agent.py`: custom tool calling a REST API

**Model:** `openai/gpt-oss-120b` on Groq · **Tool:** custom `get_weather`

### Defining a tool
```python
@tool
def get_weather(location: str) -> dict:
    """Get the current weather for a city or location.

    Args:
        location: City name, for example Bengaluru or Srinagar.
    """
```
- The `@tool` decorator turns a Python function into an LLM tool.
- The **function name**, **type hints**, and **docstring** become the tool schema the model sees, so write them clearly.

### What the tool does
1. Reads `WEATHERSTACK_API_KEY` and returns an error dict if it is missing.
2. Calls `GET https://api.weatherstack.com/current` with `access_key`, `query`, `units=m` (metric) and a 15 s timeout.
3. Handles both HTTP errors (`raise_for_status`) and API-level errors (an `"error"` key in the JSON).
4. Returns a **small, clean dict**: location, temperature, feels-like, condition, humidity, wind speed, observation time.

### Design points
- **Return errors instead of raising them.** The LLM can read the error and explain it to the user instead of the agent crashing.
- **Trim the API response** to only the fields the model needs. This uses fewer tokens and leaves less to misread.
- The system prompt tells the model to use only tool data, ask for a location if none is given, report in Celsius, and never claim to give forecasts.

**Takeaway:** any API can become an agent capability: wrap it in a typed, documented function that handles its own errors.

---

## 5. `direct_agent_with_streamlit.py`: streaming chat UI

**Model:** `openai/gpt-oss-120b` on Groq · **Tool:** Tavily (3 results, custom description)

Run with: `streamlit run single_ai_agent/direct_agent_with_streamlit.py`

### Key techniques
| Technique | Code | Why |
|---|---|---|
| Cache the agent | `@st.cache_resource` on `create_research_agent()` | Streamlit re-runs the whole script on every interaction. Caching builds the agent only once. |
| Better tool description | `search_tool.description = "... MUST provide a non-empty 'query' ..."` | Stops the model calling the tool with empty or missing arguments. |
| Chat history in the UI | `st.session_state.messages` + `st.chat_message(...)` | Keeps earlier messages on screen across re-runs. |
| Chat input | `if user_input := st.chat_input(...)` | Walrus operator: assign and check in one step. |
| Live streaming | `st.empty()` placeholder updated with `full_response + "▌"` | Typewriter effect with a cursor. |
| Filter the stream | skip if `metadata["langgraph_node"] != "model"` or `token.tool_calls` | Shows only the model's final text, not tool calls or raw tool output. |
| Error handling | `try / except` → `st.error(...)` | Shows failures in the UI instead of a stack trace. |

### Limitation to know
Only the **current** user message is sent to the agent (`inputs = {"messages": [{"role": "user", "content": user_input}]}`). The UI shows the history, but the **agent has no conversational memory**, so follow-up questions like "tell me more about that" lose context. To fix this, pass the whole `st.session_state.messages` list as `messages`, or add a LangGraph checkpointer with a `thread_id`.

---

## Model comparison

| Script | Groq model | Tools | Mode |
|---|---|---|---|
| `agent_executor_mode.py` | `qwen/qwen3.8-27b` | Tavily ×2 | invoke, verbose trace |
| `direct_agent.py` | `openai/gpt-oss-120b` | Tavily ×2 | invoke |
| `direct_agent_response_streaming.py` | `openai/gpt-oss-120b` | Tavily ×2 | stream (terminal) |
| `weather_agent.py` | `openai/gpt-oss-120b` | `get_weather` | invoke |
| `direct_agent_with_streamlit.py` | `openai/gpt-oss-120b` | Tavily ×3 | stream (web UI) |

---

## Cheat sheet

```python
# Modern agent in ~10 lines
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()
agent = create_agent(
    model=ChatGroq(model="openai/gpt-oss-120b", temperature=0),
    tools=[TavilySearch(max_results=2)],
    system_prompt="You are a careful research assistant. Cite sources.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "Your question"}]})
print(result["messages"][-1].content)
```

## Possible next steps
- Give the Streamlit agent conversation memory (see the limitation in section 5).
- Combine the search and weather tools in one multi-tool agent.
- Let the CLI scripts read the question from `input()` or `sys.argv` instead of a hard-coded string.
- Remove unused dependencies from `requirements.txt` (`duckduckgo-search` and `langchainhub` are not imported anywhere) and add `langchain-classic` explicitly.
