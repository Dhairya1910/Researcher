import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Backend.Services.Agent_Utils import Agent

st.set_page_config(page_title="AI Chatbot", layout="wide")

# -------------------------
# SESSION STATE INITIALIZE
# -------------------------
def init_state():
    defaults = {
        "menu_open": False,
        "model": "mistral-medium-latest",
        "mode": "General",
        "uploaded_file": None,
        "file_type": "text",
        "messages": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# -------------------------
# STYLES
# -------------------------
st.markdown("""
<style>

.menu-panel{
    backdrop-filter: blur(20px);
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding:20px;
    border:1px solid rgba(255,255,255,0.15);
    margin-bottom:20px;
}

.chat-bubble-user{
    background:#2b313e;
    padding:12px 18px;
    border-radius:12px;
    margin:10px 0;
    color:white;
    animation:fadeIn 0.3s ease-in;
}

.chat-bubble-ai{
    background:#1e2530;
    padding:12px 18px;
    border-radius:12px;
    margin:10px 0;
    color:white;
    animation:fadeIn 0.3s ease-in;
}

@keyframes fadeIn{
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

</style>
""", unsafe_allow_html=True)


# -------------------------
# HEADER
# -------------------------
st.title("AI Research Chatbot")
agent = Agent()



# -------------------------
# HAMBURGER MENU
# -------------------------
col1, col2 = st.columns([1,20])

with col1:
    if st.button("☰"):
        st.session_state.menu_open = not st.session_state.menu_open


# -------------------------
# MENU PANEL
# -------------------------
if st.session_state.menu_open:

    with st.container():

        st.markdown('<div class="menu-panel">', unsafe_allow_html=True)

        st.subheader("Models")

        st.session_state.model = st.radio(
            "Select Model",
            [
                "mistral-medium-latest",
                "mistral-small-latest",
                "mistral-large-latest"
            ],
            horizontal=True
        )

        st.markdown("---")

        st.subheader("Mode")

        st.session_state.mode = st.radio(
            "Select Agent mode",
            ["General","Research","Websearch"],
            horizontal=True
        )
        agent = Agent(agent_role=st.session_state.mode)
        st.markdown("---")

        st.subheader("Upload File")

        uploaded = st.file_uploader(
            "Upload PDF / DOC / Image",
            type=["pdf","docx","png","jpg","jpeg"],
            accept_multiple_files=False
        )

        if uploaded is not None:

            st.session_state.uploaded_file = uploaded
            uploaded_pdf = st.session_state.uploaded_file
            mime = uploaded.type

            if "image" in mime:
                st.session_state.file_type = "image"
                # encode the complete pdf into base64 embed and store it into vectordb

            elif "pdf" in mime:
                st.session_state.file_type = "pdf"
                text = agent.convert_and_store_to_vect_db(uploaded_pdf.getvalue())


            elif "word" in mime or "docx" in mime:
                st.session_state.file_type = "doc"
                # encode the complete pdf into base64 embed and store it into vectordb
        else:
            st.session_state.file_type = "text"

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# DISPLAY CHAT
# -------------------------
for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble-user">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="chat-bubble-ai">{msg["content"]}</div>',
            unsafe_allow_html=True
        )


# -------------------------
# CHAT INPUT
# -------------------------
prompt = st.chat_input("Ask something...")

Model_name = st.session_state.model
uploaded_pdf = st.session_state.uploaded_file 
mode = st.session_state.mode
file_type = st.session_state.file_type

if file_type is None:
    file_type = "text"
    

if prompt:
    st.session_state.messages.append({
        "role":"user",
        "content":prompt,
    })

    output = agent.Invoke_agent(prompt,file_type)

    response = f"""Jarvis : {output}"""

    st.session_state.messages.append({
        "role":"Jarvis",
        "content":response,
    })
    
    st.rerun()