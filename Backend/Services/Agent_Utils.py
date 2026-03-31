from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from chromadb.config import Settings
import sys
import os
from pathlib import Path
from typing import Generator

from Backend.Services.tools import ResearchToolkit
from Backend.Services.ResearchGraph import (
    build_research_graph,
    build_general_graph,
    build_document_graph,
)

from pypdf import PdfReader
import io
from dotenv import load_dotenv

import logging

# ──────────────────────────────────────────────────────────────
# Logger - inherits handlers from root (configured in server.py)
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def _flush():
    """Flush all root logger handlers."""
    for handler in logging.getLogger().handlers:
        handler.flush()


# ──────────────────────────────────────────────────────────────
# Module-level startup log
# ──────────────────────────────────────────────────────────────
logger.info("[Agent_Utils] Module loading...")
_flush()

BASE_DIR = Path(__file__).resolve().parent.parent

env_paths = [
    Path(__file__).resolve().parent / ".env",
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"[Agent_Utils] .env loaded from: {env_path}")
        _flush()
        env_loaded = True
        break

if not env_loaded:
    logger.warning(
        "[Agent_Utils] No .env file found — using system environment variables"
    )
    _flush()

if os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_URL"):
    _VECTOR_STORE_DIR = "/tmp/vector_datastore"
else:
    _VECTOR_STORE_DIR = str(BASE_DIR / "vector_datastore")

os.makedirs(_VECTOR_STORE_DIR, exist_ok=True)
sys.path.append(str(BASE_DIR))

logger.info(f"[Agent_Utils] Vector store dir: {_VECTOR_STORE_DIR}")
_flush()


class Agent:
    """
    Main Agent Class
    """

    def __init__(
        self,
        model_name: str = "mistral-small-latest",
        temperature: float = 0.5,
        agent_role: str = "General",
        has_document: bool = False,
    ):
        logger.info("─" * 50)
        logger.info(f"[Agent.__init__] Initializing Agent")
        logger.info(f"[Agent.__init__] model={model_name} | role={agent_role} | has_document={has_document}")
        logger.info("─" * 50)
        _flush()

        self.model_name = model_name
        self.temperature = temperature
        self.agent_role = agent_role.strip().title()
        self.has_document = has_document

        self.StringParser = StrOutputParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200
        )

        mistral_key = os.getenv("MISTRAL_API_KEY") or os.getenv("API_KEY")

        if mistral_key:
            os.environ["MISTRAL_API_KEY"] = mistral_key
            logger.info("[Agent.__init__] Mistral API Key detected ✓")
            _flush()
        else:
            logger.error("[Agent.__init__] MISTRAL_API_KEY not found!")
            _flush()
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables. "
                "Set it in your .env file (local) or Render dashboard (production)."
            )

        self.mistral_key = mistral_key

        logger.info("[Agent.__init__] Creating embedding model (mistral-embed)...")
        _flush()

        self.embedding_model = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=mistral_key,
        )
        logger.info("[Agent.__init__] Embedding model ready ✓")
        _flush()

        logger.info(
            f"[Agent.__init__] Creating Chroma vector store at: {_VECTOR_STORE_DIR}"
        )
        _flush()

        self.vector_store = Chroma(
            collection_name="User_data",
            embedding_function=self.embedding_model,
            persist_directory=_VECTOR_STORE_DIR,
            client_settings=Settings(allow_reset=True),
        )
        logger.info("[Agent.__init__] Vector store ready ✓")
        _flush()

        logger.info("[Agent.__init__] Creating ResearchToolkit...")
        _flush()

        self.toolkit = ResearchToolkit(vector_store=self.vector_store)
        logger.info("[Agent.__init__] Toolkit ready ✓")
        _flush()

        logger.info("[Agent.__init__] Building LangGraph...")
        _flush()

        self._graph = self._build_graph()

        logger.info("─" * 50)
        logger.info(f"[Agent.__init__] AGENT READY ✓")
        logger.info(f"[Agent.__init__] role={self.agent_role} | model={model_name}")
        logger.info("─" * 50)
        _flush()

    def _build_graph(self):
        """Build the appropriate graph based on agent role and document status."""
        logger.info(f"[Agent._build_graph] Building graph for role={self.agent_role}, has_document={self.has_document}")
        _flush()

        # If document is uploaded, use document-focused graph regardless of role
        if self.has_document:
            graph = build_document_graph(
                toolkit=self.toolkit,
                agent_role="Document",
            )
            logger.info(f"[Agent._build_graph] Using DOCUMENT graph (document uploaded)")
        elif self.agent_role == "Research":
            graph = build_research_graph(
                toolkit=self.toolkit,
                agent_role=self.agent_role,
            )
            logger.info(f"[Agent._build_graph] Using RESEARCH graph")
        else:
            graph = build_general_graph(
                toolkit=self.toolkit,
                agent_role=self.agent_role,
            )
            logger.info(f"[Agent._build_graph] Using GENERAL graph")

        logger.info(f"[Agent._build_graph] Graph built successfully ✓")
        _flush()
        return graph

    def convert_and_store_to_vect_db(self, bytes_data: bytes) -> bool:
        """Read and store PDF documents in vector store."""
        logger.info("[Agent.convert_and_store_to_vect_db] Processing PDF...")
        _flush()

        reader = PdfReader(io.BytesIO(bytes_data))
        logger.info(
            f"[Agent.convert_and_store_to_vect_db] PDF has {len(reader.pages)} pages"
        )
        _flush()

        self.vector_store = Chroma(
            collection_name="User_data",
            embedding_function=MistralAIEmbeddings(
                model="mistral-embed",
                api_key=self.mistral_key,
            ),
            persist_directory=_VECTOR_STORE_DIR,
            client_settings=Settings(allow_reset=True),
        )

        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        if not text.strip():
            logger.error(
                "[Agent.convert_and_store_to_vect_db] No text extracted from PDF!"
            )
            _flush()
            raise ValueError(
                "Could not extract text from PDF. "
                "The file may be image-based or corrupted."
            )

        logger.info(
            f"[Agent.convert_and_store_to_vect_db] Extracted {len(text)} characters"
        )
        _flush()

        chunks = self.splitter.split_text(text)
        logger.info(
            f"[Agent.convert_and_store_to_vect_db] Split into {len(chunks)} chunks"
        )
        _flush()

        logger.info(
            "[Agent.convert_and_store_to_vect_db] Embedding and storing chunks..."
        )
        _flush()

        self.vector_store.add_texts(chunks)

        logger.info(
            "[Agent.convert_and_store_to_vect_db] Chunks stored ✓ | Rebuilding toolkit and graph..."
        )
        _flush()

        self.has_document = True
        self.toolkit = ResearchToolkit(vector_store=self.vector_store)
        self._graph = self._build_graph()

        logger.info(
            f"[Agent.convert_and_store_to_vect_db] DONE ✓ | {len(chunks)} chunks stored | Switched to DOCUMENT graph"
        )
        _flush()
        return True

    def Invoke_agent(
        self,
        input_text: str,
        input_type: str = "text",
    ) -> Generator[str, None, None]:
        """
        Run LangGraph pipeline and yield chunks in real-time for smooth streaming.

        Args:
            input_text : User message string
            input_type : "pdf" | "docs" | "text" | "image" | "general"
        """
        logger.info("─" * 50)
        logger.info("[Agent.Invoke_agent] START (STREAMING MODE)")
        logger.info(f"[Agent.Invoke_agent] input_type : {input_type}")
        logger.info(f"[Agent.Invoke_agent] agent_role : {self.agent_role}")
        logger.info(
            f"[Agent.Invoke_agent] query      : '{input_text[:100]}{'...' if len(input_text) > 100 else ''}'"
        )
        logger.info("─" * 50)
        _flush()

        if input_type in ("pdf", "docs"):
            normalized_type = input_type
        else:
            normalized_type = "general"

        initial_state = {
            "user_query": input_text,
            "input_type": normalized_type,
            "agent_role": self.agent_role,
            "query_score": 0,
            "refined_queries": [],
            "tool_results": [],
            "context": "",
            "output": "",
        }

        logger.info(f"[Agent.Invoke_agent] Normalized input_type: {normalized_type}")
        logger.info("[Agent.Invoke_agent] Invoking LangGraph...")
        _flush()

        try:
            final_state = self._graph.invoke(initial_state)
            output = final_state.get("output", "")

            logger.info(
                f"[Agent.Invoke_agent] Graph finished | output={len(output)} chars"
            )
            _flush()

            if not output:
                logger.warning("[Agent.Invoke_agent] Empty output from graph!")
                _flush()
                yield "I was unable to generate a response. Please try again."
                return

            # Real-time streaming: yield smaller chunks more frequently for smooth animation
            # Characters per chunk - smaller = smoother but more overhead
            CHUNK_SIZE = 30  # Small chunks for smooth real-time feel
            chunk_count = 0
            
            for i in range(0, len(output), CHUNK_SIZE):
                chunk_count += 1
                chunk = output[i : i + CHUNK_SIZE]
                yield chunk
                # No artificial delay here - let the frontend handle display timing

            logger.info(f"[Agent.Invoke_agent] Streamed {chunk_count} chunks ✓")
            _flush()

        except Exception as e:
            logger.error(f"[Agent.Invoke_agent] Graph ERROR: {e}", exc_info=True)
            _flush()
            yield f"An error occurred during processing: {str(e)}"

    def clear_vectorstore(self) -> None:
        """Clear Chroma collection and reinitialize."""
        logger.info("[Agent.clear_vectorstore] Clearing vector store...")
        _flush()

        if hasattr(self, "vector_store") and self.vector_store is not None:
            try:
                all_ids = self.vector_store.get(include=[])["ids"]
                if all_ids:
                    self.vector_store.delete(ids=all_ids)
                    logger.info(
                        f"[Agent.clear_vectorstore] Deleted {len(all_ids)} entries"
                    )
                else:
                    logger.info("[Agent.clear_vectorstore] Vector store already empty")
                _flush()
            except Exception as e:
                logger.warning(
                    f"[Agent.clear_vectorstore] delete(ids) failed, using delete_collection(): {e}"
                )
                self.vector_store.delete_collection()
                _flush()

            self.vector_store = Chroma(
                collection_name="User_data",
                embedding_function=self.embedding_model,
                persist_directory=_VECTOR_STORE_DIR,
                client_settings=Settings(allow_reset=True),
            )

            self.has_document = False
            self.toolkit = ResearchToolkit(vector_store=self.vector_store)
            self._graph = self._build_graph()

            logger.info(
                "[Agent.clear_vectorstore] Vector store reinitialized | graph rebuilt ✓ | Switched back to original graph"
            )
            _flush()
