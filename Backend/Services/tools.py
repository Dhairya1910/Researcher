from langchain_core.tools import tool, BaseTool, BaseToolkit

# from langchain_community.utilities import jina_search
from langchain_community.tools import DuckDuckGoSearchRun
from exa_py import Exa
from dotenv import load_dotenv
import os
from typing import Any, List


class ResearchToolkit(BaseToolkit):
    """
    Tool kit containing all the necessary tools for agent to function.
    """

    vector_store: Any

    class Config:
        arbitrary_types_allowed = (
            True  # allows access to objects that are not known to pydantic.
        )

    def __init__(self, vector_store: Any, **kwargs):
        super().__init__(vector_store=vector_store, **kwargs)

    def get_tools(self) -> list[BaseTool]:
        load_dotenv()
        api_key = os.getenv("EXA_API_KEY")
        exa = Exa(api_key=api_key)
        duck_search = DuckDuckGoSearchRun()

        @tool
        def document_retrieval_tool(queries: List[str]) -> str:
            """
            Retrieves specific, relevant content from the provided document(s) based on a semantic search.
            You can pass multiple distinct search queries in a single tool call to batch process them.

            Args:
                queries (List[str]): A list of up to 3 optimized search strings or keyword clusters.
                                     CRITICAL: Do NOT call this tool multiple times. If you have multiple
                                     distinct concepts to look up (e.g., general NN concepts vs CNN specifics),
                                     put them all in this single list as separate strings.

            Returns:
                str: A single concatenated string containing the raw text of the relevant document chunks,
                     separated by double newlines. If retrieval fails, it returns an error message.
            """
            print(f"Document-retrieval-tool accessed with queries: {queries}")
            retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
            all_retrieved_docs = []
            seen_content = set()
            for query in queries:
                docs = retriever.invoke(query)
                for doc in docs:
                    if doc.page_content not in seen_content:
                        seen_content.add(doc.page_content)
                        all_retrieved_docs.append(doc)

            if not all_retrieved_docs:
                return "System Note: No relevant information found in the document for these queries."

            context_text = "\n\n".join(doc.page_content for doc in all_retrieved_docs)
            return context_text

        @tool
        def general_search_mode(query: str) -> str:
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

        return [general_search_mode, Advance_Search_mode, document_retrieval_tool]
