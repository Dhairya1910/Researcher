from langchain_core.tools import tool, BaseTool, BaseToolkit

# from langchain_community.utilities import jina_search
from langchain_community.tools import DuckDuckGoSearchRun
from exa_py import Exa
from dotenv import load_dotenv
import os


class ResearchToolkit(BaseToolkit):
    """
    Tool kit containing all the necessary tools for agent to function.
    """

    class Config:
        arbitrary_types_allowed = (
            True  # allows access to objects that are not known to pydantic.
        )

    def get_tools(self) -> list[BaseTool]:
        load_dotenv()
        api_key = os.getenv("EXA_API_KEY")
        exa = Exa(api_key=api_key)
        duck_search = DuckDuckGoSearchRun()

        @tool
        def general_search_mode(query: str) -> str:
            # Remark : this tool is not required. get a better prompt
            """
            Acts as a fallback retrieval tool. Use this ONLY when information
            cannot be found within the provided documents or internal context.
            It is ideal for verifying external facts, dates, or general knowledge
            that is missing from the user's uploaded files.
            """
            print("General-search-tool accessed.")
            results = duck_search.invoke(query)
            return results

        @tool
        def Advance_Search_mode(query: str) -> dict:
            """
            Executes a high-intensity, 'Deep Research' web search.

            TRIGGER THIS TOOL ONLY WHEN:
            1. The user requires 'Latest News', 'Current Events', or 'Real-time Data' (Post-2024).
            2. The internal knowledge base lacks specific technical details or external 'Source Links'.
            3. The query demands 'Deep Insights' or 'Fact-Checking' against authoritative web sources.
            4. You need to provide verifiable URLs and citations for professional-grade research.

            DO NOT USE for:
            - Basic definitions found in your pre-trained knowledge.
            - Simple conversational tasks or math.

            Returns: A list of dicts containing Title (T), URL (U), and extracted Highlights (C).
            """
            print("Advance-search-tool accessed.")

            result = exa.search(
                query,
                type="auto",
                num_results=5,
                contents={"highlights": {"max_characters": 1500}},
            )
            cleaned_data = []

            for article in result.results:
                complete_highlight = " ".join(article.highlights).strip()
                cleaned_data.append(
                    {"T": article.title, "U": article.url, "C": complete_highlight}
                )

            return cleaned_data

        return [general_search_mode, Advance_Search_mode]
