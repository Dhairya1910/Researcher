from langchain_core.tools import tool, BaseTool, BaseToolkit
from langchain_community.utilities import jina_search
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
import json


class ResearchToolkit(BaseToolkit):
    """
    Tool kit containing all the necessary tools for agent to function.
    """

    class Config:
        arbitrary_types_allowed = (
            True  # allows access to objects that are not known to pydantic.
        )

    def get_tools(self) -> list[BaseTool]:

        # creating a citation model for generating its citations.
        citation_summarization_model = ChatMistralAI(
            model="mistral-small-latest", temperature=0.8
        )

        citation_prompt = PromptTemplate(
            template="""
        You are an expert researcher and technical content writer.

        Your task is to generate a **clear, detailed, and well-structured summary** of the provided content, enriched with **accurate IEEE-style citations**.

        ==================== INPUT ====================

        Content:
        {text_content}

        Citation Sources:
        {Description_links}

        Relevant Sources links : 
        {Source_links}

        ==================== INSTRUCTIONS ====================

        1. Carefully analyze both:
        - The main Content
        - The Citation Sources (links / references)
        - Relevant Sources links (contains links)

        2. Generate a **comprehensive, point-wise summary** that:
        - Explains key ideas clearly
        - Adds useful context where relevant
        - Avoids repetition or fluff

        ==================== CITATION RULES ====================

        - Every factual statement MUST include a citation.
        - Use IEEE citation style.

        INLINE FORMAT:
        - Place citation numbers like [1], [2], [3] immediately after the sentence.
        - Number citations in order of first appearance.

        REFERENCE FORMAT:
        [N] Author(s), "Title," Source/Journal, vol. X, no. X, pp. XX–XX, Year.
        - Convert links into proper reference entries.
        - Ensure each reference includes a **clickable hyperlink**.

        ==================== OUTPUT FORMAT ====================

        Summary:
        --------
        <Structured, point-wise detailed summary with inline citations>

        References:
        -----------
        [1] ...
        [2] ...
        [3] ...
        """
        )

        output_parser = StrOutputParser()

        # used for searching
        Search = jina_search.JinaSearchAPIWrapper()

        @tool
        def general_knowledge_mode(query: str) -> str:
            # Remark : this tool is not required. get a better prompt
            """
            Answer general knowledge questions not related to uploaded documents.
            """
            print("General knowledge tool accessed.")
            return f"Answering from General knowledge {query}"

        @tool
        def Search_mode(query: str) -> str:
            # Remark : no need for llm call think alternative, just searching and getting the output and sending it back to agent.
            """
            Use this tool to gain latest news and insights about Document or user query.
            """
            print("Search tool accessed.")

            raw_data = Search.run(query)
            results = json.loads(raw_data)

            Text_content = "\n\n".join(
                f"{item.get('snippet', '')}" for item in results if item.get("snippet")
            )

            Description = "\n\n".join(
                f"{item.get('content'),''}" for item in results if item.get("content")
            )

            Links = "\n\n".join(
                f"{item.get('link','')}" for item in results if item.get("link")
            )

            limit = min(len(Description), 200000) - 1

            chain = citation_prompt | citation_summarization_model | output_parser

            output = chain.invoke(
                {
                    "text_content": Text_content,
                    "Description_links": Description[limit],
                    "Source_links": Links,
                }
            )

            return output

        return [general_knowledge_mode]
