# Researcher: Advanced AI-Powered Assistant

An intelligent, production-ready research assistant built on LangGraph and FastAPI. Researcher combines local document retrieval with live web search to deliver comprehensive, verified answers through a modern, streaming-enabled interface.

## Project Overview

Researcher is an advanced AI-powered assistant that leverages LangGraph for structured reasoning and multi-modal search capabilities. It processes user queries through a stateful reasoning engine that optimizes queries, decides when to use local vs. web sources, and verifies answers before delivery.

The system consists of three primary layers:
- Backend Agent Layer: LangGraph-based state machines for intelligent reasoning
- Integration Layer: Tool connectors for web search, document retrieval, and data processing
- Frontend Layer: FastAPI bridge and interactive web UI with real-time streaming

## Directory Structure

```
Researcher/
├── Backend/
│   ├── Services/
│   │   ├── ResearchGraph.py          # State machine definitions for research modes
│   │   ├── GraphNodes.py             # Logic gates and node implementations
│   │   ├── GraphState.py             # State structure and management
│   │   ├── tools.py                  # External tool integrations
│   │   └── prompts.py                # System prompts and instructions
│   ├── vector_datastore/             # ChromaDB vector storage (gitignored)
│   └── __init__.py
├── Frontend/
│   ├── server.py                     # FastAPI bridge server
│   ├── index.html                    # Primary UI with glassmorphic design
│   ├── main-streamlit.py             # Legacy Streamlit interface
│   └── static/                       # CSS, JS, and asset files
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project configuration (uv)
├── uv.lock                           # Locked dependency versions
├── vercel.json                       # Vercel deployment config
├── render.yaml                       # Render deployment config
├── .env.example                      # Example environment variables
├── .gitignore                        # Git exclusions
└── README.md                         # This file
```

## Key Components

### Backend Services (Backend/Services/)

**ResearchGraph.py**
Defines the core state machines for different research modes:
- General Mode: Straightforward question answering with web search
- Research Mode: In-depth investigation with multi-source validation
- Document Mode: Focused analysis of local document collections

**GraphNodes.py**
Implements specialized logic nodes:
- Query Optimization: Refines user queries for better search results
- Tool Decision: Determines whether to use local docs, web search, or both
- Answer Generation: Synthesizes information into structured responses
- Verification: Cross-checks answers for accuracy and completeness

**GraphState.py**
Defines the state structure that flows through the graph:
- User input and context
- Search results and document references
- Reasoning history and decision trails
- Final answer and confidence metrics

**tools.py**
Integrates external and internal tools:
- DuckDuckGo: Free web search without API keys
- Exa AI: Advanced semantic web search with real-time indexing
- ChromaDB: Local vector database for document retrieval
- Web Fetch: Direct content extraction from URLs

**prompts.py**
Contains system prompts that guide AI behavior:
- Research instructions for different modes
- Output formatting requirements
- Citation and source attribution rules
- Quality and verification standards

### Frontend Layer (Frontend/)

**server.py**
FastAPI application that:
- Exposes REST endpoints for research queries
- Manages WebSocket/SSE connections for streaming responses
- Bridges user requests to backend graph execution
- Handles authentication and rate limiting

**index.html**
Modern interactive UI featuring:
- Glassmorphic design with real-time feedback
- Server-Sent Events (SSE) for streaming response updates
- Query mode selection (General/Research/Document)
- Citation display and source tracking
- Responsive layout for desktop and mobile

**main-streamlit.py**
Legacy prototype interface for quick testing and iteration. Useful for rapid development cycles.

### Configuration Files

**requirements.txt**
Core Python dependencies:
- mistral-client: LLM inference
- langgraph: State machine orchestration
- fastapi: Web server framework
- chromadb: Vector database
- pydantic: Data validation
- httpx: Async HTTP client

**pyproject.toml**
Project metadata and dependency groups:
- Production dependencies
- Development dependencies
- Tool configurations (pytest, black, ruff)

**uv.lock**
Locked dependency versions for reproducible builds across environments.

## Features

### Hybrid Search Capability
Researcher combines two complementary search strategies:
- Local Search: Vector similarity search across uploaded/ingested documents via ChromaDB
- Web Search: Real-time internet search using DuckDuckGo or Exa AI
- Automatic Routing: Intelligently decides which sources to query based on query type

### Stateful Reasoning
Built on LangGraph for structured, transparent reasoning:
- Query Optimization: Refines inputs before search
- Multi-Source Aggregation: Combines results from multiple sources
- Verification Layer: Cross-checks information for accuracy
- Reasoning Transparency: Full decision trail available to users

### Streaming Interface
Real-time feedback as research progresses:
- Server-Sent Events (SSE) for live response generation
- Progressive content delivery reduces perceived latency
- Status indicators show research stage (searching, analyzing, writing)
- Token-by-token streaming for smooth user experience

### Research Modes
Tailored workflows for different use cases:
- General: Quick answers with single-pass search
- Research: Comprehensive investigation with multi-stage verification
- Document: Focused analysis of local document collections

### Citation and Attribution
Proper source tracking and attribution:
- Automatic citation generation from sources
- Reference links to original documents
- Transparency in information sourcing

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager (uv recommended)
- API keys for Mistral and Exa AI (optional for basic functionality)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/researcher.git
cd researcher
```

### Step 2: Create Virtual Environment

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Or using venv:
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

Using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```
MISTRAL_API_KEY=your_mistral_key_here
EXA_API_KEY=your_exa_api_key_here
OPENAI_API_KEY=optional_for_fallback
```

Leave keys blank to use free alternatives (DuckDuckGo for web search).

## Configuration

### Environment Variables

Required for full functionality:
- `MISTRAL_API_KEY`: Mistral API key for LLM inference
- `EXA_API_KEY`: Exa AI API key for advanced web search
- `OPENAI_API_KEY`: Optional, for fallback LLM provider

Optional configuration:
- `VECTOR_DB_PATH`: Path to ChromaDB storage (default: Backend/vector_datastore)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_SOURCES`: Maximum number of web sources to query (default: 5)
- `CHUNK_SIZE`: Document chunk size for vector embedding (default: 512)

### LangGraph Configuration

Modify Backend/Services/ResearchGraph.py to adjust:
- Graph topology (nodes and edges)
- Timeouts and retry logic
- Tool selection thresholds
- Output formatting rules

### FastAPI Configuration

In Frontend/server.py:
- CORS settings for cross-origin requests
- Rate limiting parameters
- Authentication middleware
- Response streaming settings

## Running the Application

### Option 1: FastAPI Frontend with Backend

```bash
cd Frontend
python server.py
```

The application will be available at http://localhost:8000

### Option 2: Streamlit Prototype

```bash
streamlit run Frontend/main-streamlit.py
```

Streamlit interface available at http://localhost:8501

### Option 3: Development Mode with Hot Reload

```bash
cd Frontend
fastapi dev server.py
```

Automatically restarts on code changes (requires fastapi[standard]).

### Health Check

**GET /health**

Returns system status:
```json
{
  "status": "healthy",
  "vector_db": "connected",
  "llm": "available"
}
```

See Frontend/server.py for complete endpoint documentation and OpenAPI spec at http://localhost:8000/docs

## Deployment

### Vercel (Frontend Bridge)

The frontend can be deployed to Vercel for high availability:

1. Connect your GitHub repository to Vercel
2. Configure environment variables in Vercel dashboard
3. Deploy: `vercel deploy`

Configuration is in vercel.json.

### Render (Backend Services)

Deploy the backend agent to Render:

1. Connect GitHub repository to Render
2. Create a new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python Frontend/server.py` 
Configuration is in render.yaml.

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Create a feature branch
2. Make changes with clear commit messages
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support and Community

For issues, questions, or suggestions:
- Open an issue on GitHub
