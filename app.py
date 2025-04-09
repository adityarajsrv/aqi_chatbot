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

# ✅ Define agent loader inside a cached function
@st.cache_resource
def init_agent():
    Google_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=Google_API_KEY)
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

# ✅ AQI lookup helper function
def get_aqi(city):
    """Fetches real-time AQI for a given city using the WAQI API."""
    url = f"https://api.waqi.info/feed/{city}/?token={WAQI_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "ok":
            aqi_value = data["data"]["aqi"]
            return f"The AQI in {city.title()} is {aqi_value}."
        else:
            error_msg = data.get("data") or "unknown error"
            return f"Sorry, I couldn't fetch the AQI for {city.title()}. ({error_msg})"
    else:
        return "Error: Unable to fetch data at the moment. Please try again later."


# ✅ Sidebar for direct AQI lookup
with st.sidebar:
    st.header("Real-Time AQI Lookup")
    city_input = st.text_input("Enter a city name for AQI lookup:")
    if st.button("Fetch AQI"):
        if city_input:
            aqi_result = get_aqi(city_input)
            st.write(aqi_result)
        else:
            st.write("Please enter a city name.")


# ✅ Chat history initialization
if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("AQI Chatbot")

# ✅ Display past chat messages
for msg in st.session_state["history"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ✅ Accept new user input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state["history"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    conversation_text = ""
    for msg in st.session_state["history"]:
        conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    today_date = datetime.today().strftime("%d-%m-%Y")

    prompt_template = (
        f"You are an expert in air quality monitoring and environmental awareness, specializing in providing AQI updates and health recommendations for India. "
        f"Today's date is {today_date}. "
        "You can answer both simple and complex queries effectively. "
        "For detailed questions requiring explanation, give a comprehensive, well-structured response. "
        "Your goal is to keep the public informed about air quality, its health effects, and necessary precautions. "
        "Based on the conversation so far:\n{conversation}\n"
        "and the user's new message: {user_message}\n"
        "Provide a response that is appropriately scaled in length, ensuring clarity, accuracy, and relevance."
    )

    formatted_prompt = prompt_template.format(conversation=conversation_text, user_message=user_input)
    human_message = HumanMessage(content=formatted_prompt)

    assistant_message = st.chat_message("assistant")
    with st.spinner("Thinking..."):
        full_response = ""
        response_box = assistant_message.empty()  # 👈 placeholder for assistant message

        try:
            for chunk in agent.stream({"input": formatted_prompt}, {"configurable": {"thread_id": "default"}}):
                if isinstance(chunk, dict) and "agent" in chunk:
                    msgs = chunk["agent"].get("messages", [])
                    if msgs and hasattr(msgs[0], "content"):
                        full_response = msgs[0].content.strip()
                        response_box.write(full_response)  # 👈 this writes to Streamlit web UI
        except Exception as e:
            full_response = "Sorry, something went wrong while processing your message."
            response_box.write(full_response)

        if not full_response:
            response = llm.invoke(formatted_prompt)
            full_response = response.content.strip()
            response_box.write(full_response)

    st.session_state["history"].append({"role": "assistant", "content": full_response})

