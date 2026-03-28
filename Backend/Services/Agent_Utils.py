from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver

from chromadb.config import Settings
import sys
import os
from pathlib import Path

from Backend.Services.MiddleWares import ResearcherMiddleware
from Backend.Services.tools import ResearchToolkit

from typing import TypedDict
from pypdf import PdfReader
import io
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


env_paths = [
    Path(__file__).resolve().parent / ".env",   # Backend/Services/.env
    BASE_DIR / ".env",                           # Backend/.env
    BASE_DIR.parent / ".env",                    # Project root .env
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Local .env loaded from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("No .env file found (using system environment variables from Render)")

if os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_URL"):
    _VECTOR_STORE_DIR = "/tmp/vector_datastore"
else:
    _VECTOR_STORE_DIR = str(BASE_DIR / "vector_datastore")

os.makedirs(_VECTOR_STORE_DIR, exist_ok=True)

sys.path.append(str(BASE_DIR))


class metadata(TypedDict):
    user: str
    agent_role: str
    input_type: str


class Agent:
    """
    Parent class containing all the agent Helper functions.
    """

    def __init__(
        self, model_name="mistral-small-latest", temperature=0.5, agent_role="general"
    ):
        self.model_name = model_name
        self.StringParser = StrOutputParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200
        )

        self.agent_role = agent_role

        mistral_key = os.getenv("MISTRAL_API_KEY") or os.getenv("API_KEY")

        if mistral_key:
            # Set it so ChatMistralAI can find it automatically
            os.environ["MISTRAL_API_KEY"] = mistral_key
            print("Mistral API Key detected ✅")
        else:
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables. "
                "Set it in your .env file (local) or Render dashboard (production)."
            )

        if agent_role == "Research":
            thinking_model = ChatMistralAI(
                model=self.model_name,
                temperature=temperature,
                api_key=mistral_key,
            )
            self.model = thinking_model
        else:
            self.model = ChatMistralAI(
                model=self.model_name,
                temperature=temperature,
                api_key=mistral_key,
            )

        # Embeddings
        self.embedding_model = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=mistral_key,
        )
        self.vector_store = Chroma(
            collection_name="User_data",
            embedding_function=self.embedding_model,
            persist_directory=_VECTOR_STORE_DIR,
            client_settings=Settings(allow_reset=True),
        )

        # Toolkit
        self.toolkit = ResearchToolkit(vector_store=self.vector_store)

        # Middleware
        Add_context_middleware = ResearcherMiddleware()

        # Agent
        self.agent = create_agent(
            model=self.model,
            tools=self.toolkit.get_tools(),
            system_prompt=f"""
            You are a {agent_role}, you main goal is to assist the user as much you can,
            use tools provided to you for assisting the user.
            """,
            context_schema=metadata,
            checkpointer=InMemorySaver(),
            middleware=[Add_context_middleware],
        )

    def convert_and_store_to_vect_db(self, bytes_data):
        """
        Convert byte data into text from PDF and stores it to vector database.
        """
        reader = PdfReader(io.BytesIO(bytes_data))

        # Re-init vector store (same fix applied)
        mistral_key = os.getenv("MISTRAL_API_KEY") or os.getenv("API_KEY")
        self.vector_store = Chroma(
            collection_name="User_data",
            embedding_function=MistralAIEmbeddings(
                model="mistral-embed",
                api_key=mistral_key,
            ),
            persist_directory=_VECTOR_STORE_DIR,
            client_settings=Settings(allow_reset=True),
        )

        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        chunks = self.splitter.split_text(text)
        self.vector_store.add_texts(chunks)

        return True

    def Invoke_agent(self, input_text, input_type="text"):
        """
        Based on input_type processes the user request and invokes the agent.
        """

        stream = self.agent.stream(
            input={"messages": [HumanMessage(content=input_text)]},
            context=metadata(
                user="user", agent_role=self.agent_role, input_type=input_type
            ),
            config={"configurable": {"thread_id": "1"}},
            stream_mode="messages",
        )

        for chunk in stream:
            if isinstance(chunk, tuple):
                message, _ = chunk
            else:
                message = chunk

            if isinstance(message, AIMessageChunk):
                if hasattr(message, "content") and message.content:

                    content = message.content

                    if isinstance(content, str):
                        yield content

                    elif isinstance(content, list):
                        for item in content:

                            if not isinstance(item, dict):
                                continue

                            item_type = item.get("type")

                            if item_type == "thinking":
                                continue

                            if item_type in ["text", "output_text"]:
                                yield item.get("text", "")

                            for value in item.values():
                                if isinstance(value, list):
                                    for sub in value:
                                        if (
                                            isinstance(sub, dict)
                                            and sub.get("type") == "text"
                                        ):
                                            yield sub.get("text", "")

    def clear_vectorstore(self):
        if hasattr(self, "vector_store") and self.vector_store is not None:
            try:
                all_ids = self.vector_store.get(include=[])["ids"]
                self.vector_store.delete(ids=all_ids)
                print("Whole vector store reset.")
            except Exception as e:
                self.vector_store.delete_collection()
                print(f"Collection cleared: {e}")