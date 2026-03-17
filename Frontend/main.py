import streamlit as st
import sys
from pathlib import Path

# Adjusting path for Backend imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Backend.Services.Agent_Utils import Agent

st.set_page_config(page_title="Research AI", layout="wide")


# -------------------------
# SESSION STATE INITIALIZE
# -------------------------
def init_state():
    defaults = {
        "model": "mistral-medium-latest",
        "mode": "General",
        "uploaded_file": None,
        "file_type": "text",
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# -------------------------
# MINIMALIST UI STYLING
# -------------------------
st.markdown(
    """
<style>
    /* Global App Background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Sidebar - Crisp & Professional */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Input field text color fix */
    .stSelectbox label, .stRadio label, .stFileUploader label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Chat Bubbles - High Contrast */
    .chat-bubble-user {
        background-color: #2563eb;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 2px 18px;
        margin: 10px 0;
        margin-left: auto;
        width: fit-content;
        max-width: 80%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .chat-bubble-ai {
        background-color: #ffffff;
        color: #1e293b;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 2px;
        margin: 10px 0;
        border: 1px solid #e2e8f0;
        width: fit-content;
        max-width: 80%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .user-label {
        font-size: 0.75rem;
        color: #64748b;
        text-align: right;
        margin-bottom: -5px;
    }

    .ai-label {
        font-size: 0.75rem;
        color: #2563eb;
        font-weight: bold;
        margin-bottom: -5px;
    }

    /* Title Styling */
    .main-title {
        text-align: center;
        color: #0f172a;
        font-weight: 800;
        padding: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #2563eb;'>Research AI</h2>", unsafe_allow_html=True)
    st.divider()

    st.session_state.model = st.selectbox(
        "Model Select",
        ["mistral-medium-latest", "mistral-small-latest", "mistral-large-latest"],
    )

    st.session_state.mode = st.radio("Agent Mode", ["General", "Research"])

    st.divider()

    uploaded = st.file_uploader(
        "Upload context", type=["pdf", "docx", "png", "jpg", "jpeg"]
    )

    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.info(f"Active: {uploaded.name}")

# -------------------------
# MAIN CHAT
# -------------------------
st.markdown(
    f"<h1 class='main-title'>{st.session_state.mode} Assistant</h1>",
    unsafe_allow_html=True,
)

chat_display = st.container()

with chat_display:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown('<div class="user-label">You</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="ai-label">Jarvis</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-bubble-ai">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

# -------------------------
# INPUT
# -------------------------
prompt = st.chat_input("How can I help you today?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    f_type = "text"
    if st.session_state.uploaded_file:
        if "pdf" in st.session_state.uploaded_file.type:
            f_type = "pdf"
        elif "image" in st.session_state.uploaded_file.type:
            f_type = "image"
        else:
            f_type = "doc"

    agent = Agent(agent_role=st.session_state.mode)
    with st.spinner("Writing response..."):
        output = agent.Invoke_agent(prompt, f_type)
        st.session_state.messages.append({"role": "Jarvis", "content": output})

    st.rerun()
