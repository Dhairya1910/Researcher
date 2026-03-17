from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.agents import create_agent 
from langchain.messages import HumanMessage,SystemMessage
from langchain.agents.middleware import ModelRequest,ModelResponse,AgentMiddleware
from typing import Callable,TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader
import io
from dotenv import load_dotenv


class metadata(TypedDict):
    user : str 
    agent_role : str 
    input_type : str 


"""
######################################################## Main class ########################################### 
"""

class Agent:

    """
    Parent class containing all the agent Helper functions.
    """

    def __init__(self,model_name="mistral-small-latest",temperature=0.5,agent_role='general'):

        self.StringParser = StrOutputParser()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)
        self.model_name = model_name
        self.agent_role = "Intelligent research" if agent_role == "Research" else "helpful" 

        # verify and load API keys.
        if load_dotenv():
            print("API verfied.")
        else:
            print("Api Key not verified.")

        # create Model 
        self.model = ChatMistralAI(
            model = self.model_name,
            temperature = temperature,
        )

        # Create embedding model 
        self.embedding_model = MistralAIEmbeddings(
            model = "mistral-embed"
        )

         # Create Vector datastore.
        self.vector_store = Chroma(
            collection_name = "User_data",
            embedding_function= self.embedding_model,
            persist_directory= r"d:\Researcher\Backend\vector_datastore"
        )

        # add_context middle ware
        Add_context_middleware = ResearcherMiddleware(vector_store=self.vector_store)

        # Create agent
        self.agent = create_agent(
            model = self.model,
            system_prompt= """
            You are a very {agent_role}, you main goal is to assist the user as much you can,
            use tools provided to you for assisting the user.
            """,
            context_schema =  metadata,
            middleware= [Add_context_middleware]
        )
        

    def convert_and_store_to_vect_db(self,bytes_data):
        """
        Convert byte data into text from PDF and stores it to vector database.
        """

        reader = PdfReader(io.BytesIO(bytes_data))

        text = ""
        for page in reader.pages:
            text += page.extract_text()

        chunks = self.splitter.split_text(text)
        self.vector_store.add_texts(chunks)
        return text 


    def Invoke_agent(self,input_text,input_type ="text"):
        """
        Based on input_type processes the user request and invokes the agent.
        """

        if input_type == "text":
            response = self.agent.invoke(
                    {
                        "messages" : [HumanMessage(content = input_text)]
                    },
                    context= metadata(user="user",agent_role=self.agent_role,input_type=input_type),
                ) 

        elif input_type == "pdf":
            response  = self.agent.invoke(
                {
                    "messages" : [HumanMessage(content = input_text)]
                },
                context= metadata(user="user",agent_role=self.agent_role,input_type=input_type),
            ) 
        output = response["messages"][-1].content
        output = self.StringParser.invoke(output)
        return output


    """
    ####################################### MIDDLEWARE and TOOLS Section #######################################
    """

class ResearcherMiddleware(AgentMiddleware):

    def __init__(self,vector_store):
        self.vector_store = vector_store 

    def wrap_model_call(self,request : ModelRequest, handler : Callable[[ModelRequest],ModelResponse]) -> ModelResponse:
        """
        Middleware that will add dynamic context based on agent_role and input_type of data.
        """
        user_message = request.messages[-1].content
        agent_role = request.runtime.context['agent_role']
        input_type = request.runtime.context['input_type']

        if input_type in ["pdf","docs"]: 
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            )
            retrieved_docs = retriever.invoke(user_message)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            system_prompt = f"""
                You are a {agent_role} assistant.

                Context from uploaded document:
                {context_text}
                user query : 
                {user_message}

                Rules:
                1. If the query is unrelated to the document you must say:
                "Provided document does not contain information about this query."
                2. Provide a very detailed explanation.
                3. Answer in point-by-point format.
                """

            request.messages.insert(
                0,
                SystemMessage(content=system_prompt)
            )
        else : 
            system_prompt = f"""
                You are a {agent_role} assistant.
                user query : 
                {user_message}
                Rules:
                1. Provide a very detailed explanation for given user query.
                2. Answer in point-by-point format.
                3. Try to include examples and to provide indepth explaination.
                """
            request.messages.insert(
                0,
                SystemMessage(content=system_prompt)
            )
            
        return handler(request)


        
