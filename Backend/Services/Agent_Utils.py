from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.agents import create_agent 
from langchain.messages import HumanMessage,SystemMessage
from langchain.agents.middleware import wrap_model_call,ModelRequest,ModelResponse
from typing import Callable
# from langgraph.runtime import Runtime
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader
import io
from dotenv import load_dotenv


class Agent:
    def __init__(self,model_name="mistral-small-latest",temperature=0.5,agent_role='general'):
        self.StringParser = StrOutputParser()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)
        self.model_name = model_name
        self.agent_role = "Intelligent research" if agent_role == "Research" else "helpful" 

        if load_dotenv():
            print("API verfied.")

        self.model = ChatMistralAI(
            temperature = temperature,
        )
        self.embedding_model = MistralAIEmbeddings(
            model = "mistral-embed"
        )
        self.agent = create_agent(
            model = self.model,
            system_prompt= """
            You are a very {agent_role}, you main goal is to assist the user as much you can,
            use tools provided to you for assisting the user.
            """
        )

        self.vector_store = Chroma(
            collection_name = "User_data",
            embedding_function= self.embedding_model,
            persist_directory= r"d:\Researcher\Backend\vector_datastore"
        )
    
    @wrap_model_call
    def add_context(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:

        user_message = request.messages[-1].content

        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )

        retrieved_docs = retriever.invoke(user_message)

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_prompt = f"""
            You are a {self.agent_role} assistant.

            Context from uploaded document:
            {context_text}

            Rules:
            1. If the query is unrelated to the document say:
            "Provided document does not contain information about this query."
            2. Provide a detailed explanation.
            3. Answer in point-by-point format.
            """

        request.messages.insert(
            0,
            SystemMessage(content=system_prompt)
        )

        return handler(request)


    def convert_to_base64(self,bytes_data):
        """
        Convert byte data into text from PDF.
        """
        reader = PdfReader(io.BytesIO(bytes_data))

        text = ""
        for page in reader.pages:
            text += page.extract_text()

        chunks = self.splitter.split_text(text)
        self.vector_store.add_texts(chunks)
        return text 


    def Invoke_agent(self,input_text,input_type ="text"):
        if input_type == "text":
            response = self.agent.invoke(
                    {
                        "messages" : [HumanMessage(content = input_text)]
                    }
                ) 

        elif input_type == "pdf":
            response  = self.agent.invoke(
                {
                    "messages" : [HumanMessage(content = input_text)]
                }
            ) 
        output = response["messages"][-1].content
        output = self.StringParser.invoke(output)
        return output

