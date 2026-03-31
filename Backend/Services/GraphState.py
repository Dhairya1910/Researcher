from typing import TypedDict, Literal
from pydantic import BaseModel, Field


class ResearchGraphState(TypedDict):
    """
    Central state object passed between all graph nodes.
    """

    user_query: str
    input_type: str
    agent_role: str

    classification: str

    query_score: dict
    refined_queries: list[str]

    tool_decision: dict
    tool_results: list[dict]

    context: str

    output: str


class QuerySynthesizerOutput(BaseModel):
    refined_queries: list[str] = Field(
        description="List of diverse, research-oriented sub-queries"
    )


class QueryEvaluatorOutput(BaseModel):
    query_score: dict = Field(
        description="Quality score of refined queries, integer between 0 and 10, individual for each query."
    )


class classifiyQueryOutput(BaseModel):
    result: Literal["Yes", "No"] = Field(
        description="whether casual converation or not."
    )
