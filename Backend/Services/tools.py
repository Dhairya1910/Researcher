from langchain_core.tools import tool, BaseTool, BaseToolkit

# from langchain_community.utilities import jina_search
from tavily import TavilyClient
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
        exa_api_key = os.getenv("EXA_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        exa = Exa(api_key=exa_api_key)
        tavily = TavilyClient(api_key=tavily_api_key)

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
        def general_search_mode(query: str) -> dict:
            """
            Fast, reliable web search using Tavily API for fact-checking, latest news, 
            basic definitions, and general information retrieval.
            
            Use this tool for:
            - Quick fact-checking and verification
            - Latest news and current events
            - Basic definitions and overviews
            - General information lookups
            
            Returns: Dict with search results including titles, URLs, and content snippets.
            """
            print(f"General-search-tool (Tavily) accessed with query: {query}")
            try:
                response = tavily.search(
                    query=query,
                    max_results=5,
                    search_depth="basic",
                    include_answer=True,
                    include_raw_content=False,
                )

                formatted_results = []
                for result in response.get("results", []):
                    formatted_results.append({
                        "T": result.get("title", ""),
                        "U": result.get("url", ""),
                        "C": result.get("content", ""),
                        "D": "Recent"  
                    })
                
            
                if response.get("answer"):
                    return {
                        "answer": response["answer"],
                        "sources": formatted_results
                    }
                
                return {"sources": formatted_results}
                
            except Exception as e:
                print(f"Tavily search error: {e}")
                return {
                    "error": f"Search failed: {str(e)}",
                    "sources": []
                }

        @tool
        def Advance_Search_mode(query: str) -> dict:
            """
            Executes a high-intensity, 'Deep Research' web search.

            TRIGGER THIS TOOL ONLY WHEN:
            1. The user requires 'Latest News', 'Current Events', or 'Real-time Data' (after-2024).
            2. The internal knowledge base lacks specific technical details or external 'Source Links'.
            3. The query demands 'Deep Insights' or 'Fact-Checking' against authoritative web sources.
            4. You need to provide verifiable URLs and citations for professional-grade research.

            DO NOT USE for:
            - Basic definitions found in your pre-trained knowledge.
            - Simple conversational tasks or math.

            Returns: A list of dicts containing Title (T), URL (U), Date (D), and extracted Highlights (C).
            """
            print("Advance-search-tool accessed.")

            result = exa.search(
                query,
                type="auto",
                use_autoprompt=True,
                num_results=8,
                contents={
                    "highlights": {"max_characters": 1200, "num_sentences": 5},
                    "text": {"max_characters": 2000},
                },
            )
            cleaned_data = []

            for article in result.results:
                highlights = getattr(article, "highlights", []) or []
                complete_highlight = " ".join(highlights).strip()
                # Fallback to text content if highlights empty
                if not complete_highlight:
                    complete_highlight = (getattr(article, "text", "") or "")[:1200]
                published_date = (
                    getattr(article, "published_date", None) or "Unknown date"
                )
                cleaned_data.append(
                    {
                        "T": article.title,
                        "U": article.url,
                        "D": published_date,
                        "C": complete_highlight,
                    }
                )

            return cleaned_data

        return [general_search_mode, Advance_Search_mode, document_retrieval_tool]
