# Researcher

Researcher is an advanced AI-powered research assistant designed to handle complex queries by combining local document understanding with real-time web intelligence. It enables users to perform deep, context-aware research through an interactive and modern interface.

---

## Overview

Researcher is built to assist students, developers, and researchers in exploring topics with depth and precision.

It integrates:

* Local document retrieval (PDF-based knowledge)
* Web search capabilities
* Advanced reasoning models

The system dynamically adapts its behavior depending on the selected mode, offering both general-purpose answers and expert-level analysis.

---

## Key Features

* Context-aware document analysis using vector databases
* Hybrid search combining local and web sources
* Advanced research mode for deep technical insights
* Real-time streaming responses via FastAPI (SSE)
* Modern, high-fidelity frontend with immersive UI
* Modular and extensible architecture

---

## Project Structure

```bash
Researcher/
│
├── Backend/
│   └── Services/
│       ├── Agent_Utils.py       # Core AI agent logic
│       ├── MiddleWares.py       # Prompt orchestration layer
│       └── tools.py             # Search & retrieval tools
│
├── Frontend/
│   ├── server.py               # FastAPI backend server
│   ├── index.html              # Main UI (3D/glassmorphic)
│   └── main-streamlit.py       # Legacy Streamlit UI
│
├── api/
│   └── index.py                # (THIS IS IN DEVELOPMENT STAGE)
│
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
├── uv.lock                     # Locked dependencies
├── vercel.json                 # (THIS IS IN DEVELOPMENT STAGE)
├── .python-version             # Python version (3.12.7)
├── LICENSE                     # MIT License
└── README.md
```

---

## Core Architecture

The system is composed of three main layers:

### 1. Agent Layer

Located in `Agent_Utils.py`, this is the brain of the system.

Responsibilities:

* Initialize Mistral models
* Handle user queries
* Manage vector database (ChromaDB)
* Perform document ingestion and retrieval
* Orchestrate tool usage

---

### 2. Middleware Layer

Defined in `MiddleWares.py`.

This layer dynamically modifies system prompts based on user intent.

#### Modes:

* **General Mode**

  * Clean, structured responses
  * Simplified explanations

* **Research Mode**

  * Deep technical reasoning
  * Mechanistic explanations
  * Architecture-level insights

---

### 3. Tools Layer

Defined in `tools.py`.

Provides external capabilities to the agent:

* **Document Retrieval Tool**

  * Searches uploaded PDFs using embeddings
  * Prioritized for context-aware answers

* **General Search Mode**

  * Uses DuckDuckGo for broad queries

* **Advanced Search Mode**

  * Uses Exa AI for:

    * Technical research
    * Scholarly content
    * Up-to-date information

---

## Backend (FastAPI)

The backend is implemented using FastAPI and handles:

* File uploads (PDF ingestion)
* Session state management
* Streaming responses using Server-Sent Events (SSE)

### Key File

* `server.py` → Main API layer connecting frontend and agent

---

## Frontend

### Primary UI

* `index.html`
* Designed with:

  * Glassmorphism
  * 3D depth effects
  * Cinematic transitions

### Alternative UI

* `main-streamlit.py`
* Used for rapid prototyping and debugging

---

## Workflows

### 1. Context-Aware Research

1. User uploads PDF(s)
2. Documents are:

   * Parsed
   * Chunked
   * Embedded
   * Stored in ChromaDB
3. Queries prioritize document context

---

### 2. Hybrid Search

If local data is insufficient:

* Step 1: Try document retrieval
* Step 2: Fallback to web search (DuckDuckGo)
* Step 3: Use Exa AI for deep research queries

---

### 3. Adaptive Response Modes

| Mode     | Behavior                   |
| -------- | -------------------------- |
| General  | Simple, structured answers |
| Research | Deep technical analysis    |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/researcher.git
cd researcher
```

### 2. Setup Environment

```bash
uv venv --python 3.12
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_key
EXA_API_KEY=your_key
```

---

## Running the Project

### Run FastAPI Server

```bash
uvicorn Frontend.server:app --reload
```

### Optional: Streamlit UI

```bash
streamlit run Frontend/main-streamlit.py
```

---

## Tech Stack

* **LLM**: Mistral
* **Backend**: FastAPI
* **Frontend**: HTML/CSS/JS + Streamlit
* **Vector DB**: ChromaDB
* **Search APIs**:

  * DuckDuckGo
  * Exa AI
* **Deployment**: Vercel
* **Environment Manager**: uv

---

## Current Status

* Core agent architecture implemented
* Hybrid search pipeline functional
* Streaming responses enabled
* Frontend UI developed
* Deployment pipeline configured

---

## Future Improvements

* Multi-document reasoning
* Memory-based conversations
* Citation and source tracking
* UI enhancements and responsiveness
* Fine-tuned research agents
* Collaboration features

---

## License

This project is licensed under the MIT License.

---

## Notes

* Performance depends on API limits and model behavior
* Research Mode may be slower due to deeper reasoning
* Ensure all API keys are configured before running

---
