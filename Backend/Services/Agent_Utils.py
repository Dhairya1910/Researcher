from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver

from chromadb.config import Settings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Backend.Services.MiddleWares import ResearcherMiddleware
from Backend.Services.tools import ResearchToolkit

from typing import TypedDict
from pypdf import PdfReader
import io
from dotenv import load_dotenv


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

        # verify and load API keys.
        if load_dotenv():
            print("API verfied.")
        else:
            print("Api Key not verified.")

        # create Model
        if agent_role == "Research":
            thinking_model = ChatMistralAI(
                model=self.model_name,
                # max_completion_tokens=50000,
                temperature=temperature,
            )
            # self.model = thinking_model.with_thinking_mode(enabled=True)
            self.model = thinking_model
        else:
            self.model = ChatMistralAI(
                model=self.model_name,
                temperature=temperature,
            )
        # Create embedding model
        self.embedding_model = MistralAIEmbeddings(model="mistral-embed")

        # Create Vector datastore.
        self.vector_store = Chroma(
            collection_name="User_data",
            embedding_function=self.embedding_model,
            persist_directory=r"d:\Researcher\Backend\vector_datastore",
            client_settings=Settings(allow_reset=True),
        )

        # Initiating toolkit
        self.toolkit = ResearchToolkit(vector_store=self.vector_store)

        # add_context middle ware
        Add_context_middleware = ResearcherMiddleware()

        # Create agent
        self.agent = create_agent(
            model=self.model,
            tools=self.toolkit.get_tools(),
            system_prompt="""
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
        self.vector_store = Chroma(
            collection_name="User_data",
            embedding_function=self.embedding_model,
            persist_directory=r"d:\Researcher\Backend\vector_datastore",
            client_settings=Settings(allow_reset=True),
        )

        text = ""
        for page in reader.pages:
            text += page.extract_text()

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
                # self.vector_store.reset_collection()
                print("Whole vector store reset.")
                # self.vector_store._client.create_collection("User_data")
            except Exception as e:
                self.vector_store.delete_collection()
                print(f"Collection cleared: {e}")
