from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

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

        self.StringParser = StrOutputParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200
        )
        self.model_name = model_name
        self.agent_role = (
            "Intelligent research" if agent_role == "Research" else "helpful"
        )

        self.toolkit = ResearchToolkit()

        # verify and load API keys.
        if load_dotenv():
            print("API verfied.")
        else:
            print("Api Key not verified.")

        # create Model
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

        # add_context middle ware
        Add_context_middleware = ResearcherMiddleware(vector_store=self.vector_store)

        # Create agent
        self.agent = create_agent(
            model=self.model,
            tools=self.toolkit.get_tools(),
            system_prompt="""
            You are a {agent_role}, you main goal is to assist the user as much you can,
            use tools provided to you for assisting the user.
            """,
            context_schema=metadata,
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

        response = self.agent.invoke(
            {"messages": [HumanMessage(content=input_text)]},
            context=metadata(
                user="user", agent_role=self.agent_role, input_type=input_type
            ),
        )

        output = response["messages"][-1].content
        output = self.StringParser.invoke(output)
        return output

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
