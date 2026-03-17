from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain_core.messages import SystemMessage
from typing import Callable


class ResearcherMiddleware(AgentMiddleware):
    """
    Class contains custom and prebuilt middleware that modifies agent at runtime.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """
        Middleware that will add dynamic context based on agent_role and input_type of data.
        """
        user_message = request.messages[-1].content
        agent_role = request.runtime.context["agent_role"]
        input_type = request.runtime.context["input_type"]

        if input_type in ["pdf", "docs"]:
            print("using a pdf to answer.")
            retriever = self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 5}
            )
            retrieved_docs = retriever.invoke(user_message)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            system_prompt = f"""
            You are a {agent_role} assistant with access to tools.

            UPLOADED DOCUMENT CONTEXT 
            {context_text}
            User Query: {user_message}

            STRICT RESPONSE PROTOCOL
            =========================

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
            request.messages.insert(0, SystemMessage(content=system_prompt))
        else:
            system_prompt = f"""
                You are a {agent_role} assistant and you have access to certain tools use them as per you require.
                user query : 
                {user_message}
                TOOLKIT : [general_knowledge_mode : used to generate relevent responses , Search_mode : used to search web for gaining more knowledge]

                STRICT RESPONSE PROTOCOL
                ========================
                STEP-1 (MANDATORY) : Before generating response think of which tool can be used for you TOOLKIT.
                - Use the appropriate tools to generate detailed answers.
                - you should atleast use one tool inorder to generate response.

                STEP -2 : 
                - Provide a very detailed, point-by-point explanation.
                - Include examples where helpful.
                - Answer in point-by-point format.

                IMPORTANT : NEVER SKIP STEP - 1, skipping STEP-1 will be considered as BREAK A MANDATORY PROTOCOL.  
                """
            request.messages.insert(0, SystemMessage(content=system_prompt))

        return handler(request)
