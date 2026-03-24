import streamlit as st
import sys
import time
from datetime import datetime
from pathlib import Path

# Adjusting path for Backend imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Backend.Services.Agent_Utils import Agent

st.set_page_config(page_title="Research AI", layout="wide")


"""
|============================================================|
|This file was only created for testing out working of agent.|
|============================================================|
"""

# -------------------------
# SESSION STATE INITIALIZE
# -------------------------
def init_state():
    defaults = {
        "model": "mistral-medium-latest",
        "mode": "General",
        "current_agent_mode": "General",
        "Agent": None,
        "uploaded_file": None,
        "file_stored": False,
        "file_type": "text",
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


st.markdown(
    """
<style>
.stApp { background-color: #f8fafc; }

section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}

.chat-bubble-user {
    background-color: #2563eb;
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 2px 18px;
    margin: 10px 0;
    margin-left: auto;
    max-width: 80%;
    width: fit-content;            
    word-wrap: break-word;          
    white-space: pre-wrap; 
}


.chat-bubble-ai, .chat-bubble-user {
    animation: fadeIn 0.2s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-label, .ai-label {
    font-size: 0.75rem;
    margin-bottom: -5px;
}

.user-label {
    color: #64748b;
    text-align: right;
}

.ai-label {
    color: #2563eb;
    font-weight: bold;
}

.timestamp {
    font-size: 0.65rem;
    color: #94a3b8;
    margin-top: 2px;
}

.main-title {
    text-align: center;
    font-weight: 800;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#2563eb;'>Research AI</h2>", unsafe_allow_html=True)
    st.divider()

    st.session_state.model = st.selectbox(
        "Model Select",
        [
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-large-latest",
            "magistral-medium-latest",
            "magistral-small-latest",
        ],
    )

    st.session_state.mode = st.radio("Agent Mode", ["General", "Research"])

    if st.session_state.mode == "Research":
        st.session_state.model = "magistral-medium-latest"

    st.divider()

    uploaded = st.file_uploader(
        "Upload context", type=["pdf", "docx", "png", "jpg", "jpeg"]
    )

    if uploaded is None and st.session_state.file_stored:
        if st.session_state.Agent:
            st.session_state.Agent.clear_vectorstore()
            st.session_state.file_stored = False

    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.info(f"Active: {uploaded.name}")


st.markdown(
    f"<h1 class='main-title'>{st.session_state.mode} Assistant</h1>",
    unsafe_allow_html=True,
)

# Display history
for msg in st.session_state.messages:
    label = "You" if msg["role"] == "user" else "Jarvis"
    label_class = "user-label" if msg["role"] == "user" else "ai-label"
    bubble_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-ai"

    st.markdown(f'<div class="{label_class}">{label}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="{bubble_class}">{msg["content"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="timestamp">{msg.get("time","")}</div>',
        unsafe_allow_html=True,
    )

prompt = st.chat_input("How can I help you today?")

if prompt:
    start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M")

    # Create agent if needed
    if (
        not st.session_state.Agent
        or st.session_state.mode != st.session_state.current_agent_mode
    ):
        with st.spinner("Initializing agent..."):
            st.session_state.Agent = Agent(
                model_name=st.session_state.model,
                agent_role=st.session_state.mode,
            )
            st.session_state.current_agent_mode = st.session_state.mode

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "time": timestamp}
    )

    st.markdown('<div class="user-label">You</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="chat-bubble-user">{prompt}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="timestamp">{timestamp}</div>', unsafe_allow_html=True)

    # File handling
    f_type = "text"
    if st.session_state.uploaded_file:
        if "pdf" in st.session_state.uploaded_file.type:
            f_type = "pdf"
            if not st.session_state.file_stored:
                with st.spinner("Processing PDF..."):
                    st.session_state.Agent.convert_and_store_to_vect_db(
                        st.session_state.uploaded_file.getvalue()
                    )
                    st.session_state.file_stored = True
        elif "image" in st.session_state.uploaded_file.type:
            f_type = "image"

    # Streaming response
    stream_gen = st.session_state.Agent.Invoke_agent(prompt, f_type)

    output = ""
    bubble = st.empty()

    # Thinking indicator
    bubble.markdown(
        '<div class="chat-bubble-ai">Thinking...</div>',
        unsafe_allow_html=True,
    )

    for chunk in stream_gen:
        output += chunk

        bubble.markdown(
            f'<div class="chat-bubble-ai">{output}▌</div>',
            unsafe_allow_html=True,
        )

    bubble.markdown(
        f'<div class="chat-bubble-ai">{output}</div>',
        unsafe_allow_html=True,
    )

    end_time = time.time()
    exec_time = round(end_time - start_time, 2)
    ai_timestamp = datetime.now().strftime("%H:%M")

    st.markdown(
        f'<div class="timestamp">{ai_timestamp} • {exec_time}s</div>',
        unsafe_allow_html=True,
    )

    # Save assistant response
    if output:
        st.session_state.messages.append(
            {
                "role": "Jarvis",
                "content": output,
                "time": f"{ai_timestamp} • {exec_time}s",
            }
        )
