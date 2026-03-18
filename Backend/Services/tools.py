from langchain_core.tools import tool, BaseTool, BaseToolkit
from langchain_community.utilities import jina_search
from langchain_community.tools import DuckDuckGoSearchRun
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

        Search = jina_search.JinaSearchAPIWrapper()
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
            # Remark : no need for llm call think alternative, just searching and getting the output and sending it back to agent.
            """
            Use this for high-intensity research or when the user specifically
            asks for 'latest news', 'deep insights', or 'source links' related
            to a document topic. Use this if the documents provide a base
            concept but lack the most current real-world data or external
            references (URLs).
            """

            print("Advance-search-tool accessed.")

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

            limit = min(len(Description), 10000)
            search_summary = {
                "content": Text_content,
                "Description": Description[:limit],
                "Source_links": Links,
            }
            return search_summary

        return [general_search_mode, Advance_Search_mode]
