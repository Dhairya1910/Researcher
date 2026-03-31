from langgraph.graph import StateGraph, START, END

from Backend.Services.GraphState import ResearchGraphState
from Backend.Services.GraphNodes import (
    query_synthesizer_node,
    query_evaluator_node,
    query_optimizer_node,
    check_query_quality,
    make_tool_executor_node,
    make_generate_output_node,
    make_context_extractor_node,
    tool_decision_node,
    query_classifier_node,
)


def build_research_graph(toolkit, agent_role: str = "Research"):
    workflow = StateGraph(ResearchGraphState)

    workflow.add_node("query_classifier", query_classifier_node)
    workflow.add_node("context_extractor", make_context_extractor_node(toolkit))
    workflow.add_node("query_synthesizer", query_synthesizer_node)
    workflow.add_node("query_evaluator", query_evaluator_node)
    workflow.add_node("query_optimizer", query_optimizer_node)
    workflow.add_node("tool_decision", tool_decision_node)
    workflow.add_node("tool_executor", make_tool_executor_node(toolkit))
    workflow.add_node("generate_output", make_generate_output_node(agent_role))

    workflow.add_edge(START, "query_classifier")

    workflow.add_conditional_edges(
        "query_classifier",
        lambda state: state["classification"],
        {
            "Yes": "generate_output",
            "No": "context_extractor",
        },
    )

    workflow.add_edge("context_extractor", "query_synthesizer")
    workflow.add_edge("query_synthesizer", "query_evaluator")

    workflow.add_conditional_edges(
        "query_evaluator",
        check_query_quality,
        {
            "Approved": "tool_decision",
            "Need_Improvement": "query_optimizer",
        },
    )

    workflow.add_edge("query_optimizer", "tool_decision")
    workflow.add_edge("tool_decision", "tool_executor")
    workflow.add_edge("tool_executor", "generate_output")
    workflow.add_edge("generate_output", END)

    return workflow.compile()


def build_general_graph(toolkit, agent_role: str = "General"):
    workflow = StateGraph(ResearchGraphState)

    workflow.add_node("query_classifier", query_classifier_node)
    workflow.add_node("context_extractor", make_context_extractor_node(toolkit))
    workflow.add_node("query_synthesizer", query_synthesizer_node)
    workflow.add_node("tool_decision", tool_decision_node)
    workflow.add_node("tool_executor", make_tool_executor_node(toolkit))
    workflow.add_node("generate_output", make_generate_output_node(agent_role))

    workflow.add_edge(START, "query_classifier")

    workflow.add_conditional_edges(
        "query_classifier",
        lambda state: state["classification"],
        {
            "Yes": "generate_output",
            "No": "context_extractor",
        },
    )

    workflow.add_edge("context_extractor", "query_synthesizer")
    workflow.add_edge("query_synthesizer", "tool_decision")
    workflow.add_edge("tool_decision", "tool_executor")
    workflow.add_edge("tool_executor", "generate_output")
    workflow.add_edge("generate_output", END)

    return workflow.compile()


def build_document_graph(toolkit, agent_role: str = "Document"):
    """
    Specialized graph for handling uploaded documents.
    - First attempts to answer from the document
    - If document lacks information, intelligently uses web search tools
    - Ensures both document retrieval AND general/advanced search can be accessed
    """
    workflow = StateGraph(ResearchGraphState)

    workflow.add_node("query_classifier", query_classifier_node)
    workflow.add_node("context_extractor", make_context_extractor_node(toolkit))
    workflow.add_node("query_synthesizer", query_synthesizer_node)
    workflow.add_node("tool_decision", tool_decision_node)
    workflow.add_node("tool_executor", make_tool_executor_node(toolkit))
    workflow.add_node("generate_output", make_generate_output_node(agent_role))

    workflow.add_edge(START, "query_classifier")

    workflow.add_conditional_edges(
        "query_classifier",
        lambda state: state["classification"],
        {
            "Yes": "generate_output",
            "No": "context_extractor",
        },
    )

    workflow.add_edge("context_extractor", "query_synthesizer")
    workflow.add_edge("query_synthesizer", "tool_decision")
    workflow.add_edge("tool_decision", "tool_executor")
    workflow.add_edge("tool_executor", "generate_output")
    workflow.add_edge("generate_output", END)

    return workflow.compile()
