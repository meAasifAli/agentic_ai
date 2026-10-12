import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool  # Add this

load_dotenv()

# 1. Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# 2. Create a custom WeatherStack tool
@tool
def get_weather(location: str) -> dict:
    """Get the current weather for a city or location.

    Args:
        location: City name, for example Bengaluru or Srinagar.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return {"error": "WEATHERSTACK_API_KEY is missing from .env"}

    url = "https://api.weatherstack.com/current"

    try:
        response = requests.get(
            url,
            params={
                "access_key": api_key,
                "query": location,
                "units": "m",
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return {"error": data["error"].get("info", "WeatherStack API error")}

        current = data["current"]
        loc = data["location"]

        return {
            "location": f"{loc['name']}, {loc['country']}",
            "temperature_c": current["temperature"],
            "feels_like_c": current["feelslike"],
            "condition": current["weather_descriptions"][0],
            "humidity_percent": current["humidity"],
            "wind_speed_kmh": current["wind_speed"],
            "observation_time": current["observation_time"],
        }

    except requests.RequestException as exc:
        return {"error": f"Weather request failed: {exc}"}
    
    

# 3. Register the tool
tools = [get_weather]


system_prompt = """
You are a helpful and accurate weather assistant.

Instructions:
- Use the get_weather tool whenever the user asks about current weather.
- Ask the user for a city or location if none is provided.
- Use only the weather data returned by the tool.
- Never invent temperatures, weather conditions, humidity, or wind speeds.
- Clearly mention the location and temperature in Celsius.
- Include the feels-like temperature, weather condition, humidity, and wind speed when available.
- If the tool returns an error, explain it clearly without fabricating results.
- If the requested location cannot be found, ask the user to provide a more specific location.
- Keep responses concise, friendly, and easy to understand.
- Do not claim to provide forecasts unless the tool supports them.
"""


# 4. Create the modern LangChain agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# 5. Invoke the agent
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the current weather in Bengaluru?"
        }
    ]
})

print(response["messages"][-1].content)