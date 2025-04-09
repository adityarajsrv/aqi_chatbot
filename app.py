import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

WAQI_API_KEY = os.getenv("WAQI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize agent (cached to avoid reloading on every run)
@st.cache_resource
def init_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY)
    search = DuckDuckGoSearchRun()
    tools = [search]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent, llm

agent, llm = init_agent()

# ✅ Get AQI function
def get_aqi(city):
    url = f"https://api.waqi.info/feed/{city}/?token={WAQI_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "ok":
            aqi = data["data"]["aqi"]
            return f"The AQI in {city.title()} is {aqi}."
        else:
            return f"Sorry, couldn't fetch AQI for {city.title()}."
    else:
        return "Error: Failed to fetch AQI. Please try again later."

# ✅ Sidebar: City AQI Checker
with st.sidebar:
    st.header("Real-Time AQI Lookup")
    city = st.text_input("Enter a city name")
    if st.button("Get AQI"):
        if city:
            result = get_aqi(city)
            st.write(result)
        else:
            st.warning("Please enter a city name.")

# ✅ Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("🌫️ AQI Awareness Chatbot")

# ✅ Show chat history
for msg in st.session_state["history"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ✅ Chat input
user_input = st.chat_input("Ask about AQI, pollution effects, or prevention...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state["history"].append({"role": "user", "content": user_input})

    # Build prompt
    conversation_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["history"]])
    today = datetime.today().strftime("%d-%m-%Y")
    prompt = (
        f"You are an expert AQI and environment chatbot. Date: {today}. "
        f"Help users understand air pollution, AQI, health risks, and solutions. "
        f"Conversation so far:\n{conversation_text}\n"
        f"User's latest message: {user_input}"
    )

    human_message = HumanMessage(content=prompt)
    assistant_box = st.chat_message("assistant")
    response_placeholder = assistant_box.empty()

    try:
        with st.spinner("Thinking..."):
            final_response = ""
            for chunk in agent.stream({"input": prompt}, {"configurable": {"thread_id": "default"}}):
                if isinstance(chunk, dict) and "agent" in chunk:
                    messages = chunk["agent"].get("messages", [])
                    if messages and hasattr(messages[0], "content"):
                        final_response = messages[0].content.strip()
                        response_placeholder.write(final_response)
    except Exception as e:
        final_response = "Sorry, something went wrong while processing your message."
        response_placeholder.write(final_response)

    if not final_response:
        fallback = llm.invoke(prompt)
        final_response = fallback.content.strip()
        response_placeholder.write(final_response)

    st.session_state["history"].append({"role": "assistant", "content": final_response})
