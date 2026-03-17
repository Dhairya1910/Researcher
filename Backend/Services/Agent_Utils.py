from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.agents import create_agent 
from langchain.messages import HumanMessage,SystemMessage
from langchain.agents.middleware import ModelRequest,ModelResponse,AgentMiddleware
from langchain_core.tools import tool, BaseTool, BaseToolkit
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import jina_search
from langchain_chroma import Chroma

from typing import Callable,TypedDict
# from pydantic import Field
from pypdf import PdfReader
import io
import json
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
        self.agent_role = "Intelligent research" if agent_role == "Research" else "helpful"  "Websearcher" if agent_role == "Websearch" else "helpful"

        self.toolkit = ResearchToolkit()


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
            tools = self.toolkit.get_tools(),
            system_prompt= """
            You are a {agent_role}, you main goal is to assist the user as much you can,
            use tools provided to you for assisting the user.
            """,
            context_schema =  metadata,
            middleware= [Add_context_middleware],
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
        
        response = self.agent.invoke(
                {
                    "messages" : [HumanMessage(content = input_text)]
                },
                context= metadata(user="user",agent_role=self.agent_role,input_type=input_type),
            )  

        output = response["messages"][-1].content
        output = self.StringParser.invoke(output)
        return output


    """
    ####################################### MIDDLEWARE Section #######################################
    """

class ResearcherMiddleware(AgentMiddleware):

    """
    Class contains custom and prebuilt middleware that modifies agent at runtime.
    """

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
            print("using a pdf to answer.")
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            )
            retrieved_docs = retriever.invoke(user_message)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            system_prompt = f"""
            You are a {agent_role} assistant with access to tools.

            === UPLOADED DOCUMENT CONTEXT ===
            {context_text}
            =================================

            User Query: {user_message}

            === STRICT RESPONSE PROTOCOL ===

            STEP 1 — RELEVANCE CHECK (MANDATORY):
            - Carefully check whether the user query can be answered using the DOCUMENT CONTEXT above.
            - If YES → answer directly from the document context.
            - If NO  → you MUST first say exactly:
            "The provided document does not contain information about this query."
            Then IMMEDIATELY use the [General Knowledge Tool] to answer.

            STEP 2 — ANSWERING:
            - Provide a very detailed, point-by-point explanation.
            - Include examples where helpful.
            - Never skip STEP 1 — it is required even if you use a tool.
            IMPORTANT: Do NOT silently use a tool without first completing STEP 1.
            """
            request.messages.insert(
                0,
                SystemMessage(content=system_prompt)
            )
        else : 
            system_prompt = f"""
                You are a {agent_role} assistant and you have access to certain tools use them as per you require..
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



"""
####################################### TOOLs Section #######################################
"""

class ResearchToolkit(BaseToolkit):
    """
    Tool kit containing all the necessary tools for agent to function.
    """
    class Config: 
        arbitrary_types_allowed = True # allows access to objects that are not known to pydantic.

    def get_tools(self) -> list[BaseTool]:

        # creating a citation model for generating its citations.
        citation_summarization_model = ChatMistralAI(
            model = "mistral-small-latest",
            temperature = 0.8
        ) 

        citation_prompt = PromptTemplate(
            template = """
            Content : 
            {text_content}
            Citation_content : 
            {Description_links}

            Your an expert content writer , researcher for the given Content you have to return a detailed summary. 

            RULES YOU MUST FOLLOW : 
            1. If the Links are Citation_content then create citations accordingly.
            2. You must use the Citation_content and Content to generate a more detailed summary.
            """,
        ) 

        output_parser = StrOutputParser()
        
        # used for searching 
        Search = jina_search.JinaSearchAPIWrapper()

        @tool 
        def general_knowledge_mode(query:str) -> str:
            """
            Answer general knowledge questions not related to uploaded documents.
            """
            print("General knowledge tool accessed.")
            return f"Answering from General knowledge {query}"

        @tool
        def Search_mode(query:str) -> str:
            """
            Use this tool to gain latest news and insights about Document or user query.
            """
            print("Search tool accessed.")
            raw_data = Search.run(query) 
            results = json.loads(raw_data)

            Text_content = "\n\n".join(
                    f"{item.get('snippet', '')}"
                    for item in results
                    if item.get('snippet')
                )
            
            Description_links = "\n\n".join(
                f"{item.get('content'),''}"
                for item in results
                if item.get('content')
                )
            
            limit = min(len(Description_links),200000) -1 

            chain = citation_prompt | citation_summarization_model | output_parser

            output = chain.invoke(
                {
                    "text_content" : Text_content,
                    "Description_links" : Description_links[limit]
                }
            ) 

            return output 

        return [general_knowledge_mode,Search_mode]
