# Agentic AI Course

Hands-on examples for building AI agents in Python with **LangChain**, a **Groq**-hosted LLM, and external tools (Tavily web search and the WeatherStack weather API).

Each script is a self-contained example of one idea, from a single agent call up to a streaming chat UI.

## Project structure

```
agentic_course/
├── .env                      # API keys (not committed)
├── .gitignore                # ignores venv/ and .env
├── requirements.txt          # Python dependencies
└── single_ai_agent/
    ├── NOTES.md                            # concept notes for this module
    ├── agent_executor_mode.py              # legacy ReAct agent + AgentExecutor
    ├── direct_agent.py                     # modern create_agent + Tavily search
    ├── direct_agent_response_streaming.py  # same agent, streamed token by token
    ├── direct_agent_with_streamlit.py      # streaming chat UI in Streamlit
    └── weather_agent.py                    # custom @tool calling the WeatherStack API
```

## Tech stack

| Component | Purpose |
|---|---|
| `langchain` (v1) | `create_agent`, the modern agent API (runs on LangGraph) |
| `langchain-classic` | Legacy `create_react_agent` / `AgentExecutor` (used by `agent_executor_mode.py`) |
| `langchain-groq` | `ChatGroq`, the chat model wrapper for Groq's inference API |
| `langchain-tavily` | `TavilySearch`, a web search tool built for LLM agents |
| `requests` | HTTP calls inside the custom weather tool |
| `python-dotenv` | Loads API keys from `.env` |
| `streamlit` | Chat web UI |

## Setup

1. **Create and activate a virtual environment** (Windows PowerShell):

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

   > `agent_executor_mode.py` imports from `langchain_classic`. It is currently installed as a transitive dependency of `langchain-community`. If the import fails, run `pip install langchain-classic`.

3. **Create a `.env` file** in the project root:

   ```env
   GROQ_API_KEY=your_groq_key
   TAVILY_API_KEY=your_tavily_key
   WEATHERSTACK_API_KEY=your_weatherstack_key
   ```

   | Key | Where to get it | Used by |
   |---|---|---|
   | `GROQ_API_KEY` | https://console.groq.com | every script |
   | `TAVILY_API_KEY` | https://app.tavily.com | every script except `weather_agent.py` |
   | `WEATHERSTACK_API_KEY` | https://weatherstack.com | `weather_agent.py` |

## Running the examples

Run these from the project root with the venv active:

```powershell
python single_ai_agent/direct_agent.py                     # one-shot research answer
python single_ai_agent/direct_agent_response_streaming.py  # same, streamed to the terminal
python single_ai_agent/agent_executor_mode.py              # legacy ReAct loop (verbose trace)
python single_ai_agent/weather_agent.py                    # weather in Bengaluru via custom tool
streamlit run single_ai_agent/direct_agent_with_streamlit.py  # chat UI at http://localhost:8501
```

The CLI scripts use hard-coded questions. Edit the `content` / `input` string in each file to ask something else.

## Learning path

| # | Script | Concept |
|---|---|---|
| 1 | `agent_executor_mode.py` | The classic **ReAct** loop: Thought → Action → Observation, driven by a text prompt |
| 2 | `direct_agent.py` | The modern **`create_agent`** API with a system prompt and a prebuilt tool |
| 3 | `direct_agent_response_streaming.py` | **Streaming** output with `agent.stream(..., stream_mode="messages")` |
| 4 | `weather_agent.py` | Writing a **custom tool** with the `@tool` decorator around a REST API |
| 5 | `direct_agent_with_streamlit.py` | A **chat UI** with session state, cached agent, and live streaming |

See [`single_ai_agent/NOTES.md`](single_ai_agent/NOTES.md) for detailed notes on each script.
