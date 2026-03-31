import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import json

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from Backend.Services.prompts import (
    query_synthesizer_prompt,
    query_evaluator_prompt,
    query_optimizer_prompt,
)
from Backend.Services.GraphState import (
    ResearchGraphState,
    QuerySynthesizerOutput,
    QueryEvaluatorOutput,
    classifiyQueryOutput,
)

import logging

logger = logging.getLogger(__name__)


def _flush():
    """Flush all root logger handlers."""
    for handler in logging.getLogger().handlers:
        handler.flush()


logger.info("[GraphNodes] Module loading...")
_flush()

BASE_DIR = Path(__file__).resolve().parent.parent

_env_paths = [
    Path(__file__).resolve().parent / ".env",
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
        logger.info(f"[GraphNodes] .env loaded from: {_env_path}")
        _flush()
        break

logger.info("[GraphNodes] Module ready ✓")
_flush()


def make_context_extractor_node(toolkit):
    """
    Returns a context extractor node with toolkit captured in closure.
    Extracts sample content from uploaded document for better query generation.
    """
    
    def context_extractor_node(state: ResearchGraphState) -> dict:
        """
        Extracts context from uploaded document if available.
        Gets a representative sample of document content for query synthesis.
        """
        input_type = state.get("input_type", "general")
        
        # Only extract context if document is available
        if input_type not in ("pdf", "docs"):
            logger.info("[ContextExtractor] No document available, skipping context extraction")
            _flush()
            return {"context": ""}
        
        logger.info("[ContextExtractor] Document detected, extracting context sample...")
        _flush()
        
        try:
            tools_list = toolkit.get_tools()
            tools_map = {t.name: t for t in tools_list}
            doc_tool = tools_map.get("document_retrieval_tool")
            
            if not doc_tool:
                logger.warning("[ContextExtractor] Document tool not found")
                _flush()
                return {"context": ""}
            
            # Extract sample content using broad queries
            user_query = state.get("user_query", "")
            sample_queries = [user_query, "overview", "introduction", "summary", "key concepts"]
            
            start = time.time()
            context_sample = doc_tool.invoke({"queries": sample_queries[:3]})
            elapsed = time.time() - start
            
            # Limit context size to avoid overwhelming the prompt
            if len(context_sample) > 3000:
                context_sample = context_sample[:3000] + "..."
            
            logger.info(f"[ContextExtractor] Extracted {len(context_sample)} chars in {elapsed:.2f}s")
            _flush()
            
            return {"context": context_sample}
            
        except Exception as e:
            logger.error(f"[ContextExtractor] Error extracting context: {e}", exc_info=True)
            _flush()
            return {"context": ""}
    
    return context_extractor_node


def _get_api_key() -> str:
    key = os.getenv("MISTRAL_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise ValueError(
            "MISTRAL_API_KEY not found. "
            "Set it in your .env file or Render dashboard."
        )
    return key


def _make_synthesizer_model():
    """Fast + creative"""
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.8,
        api_key=_get_api_key(),
    ).with_structured_output(QuerySynthesizerOutput)


def _classify_query_model():
    """Fast + creative"""
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.0,
        api_key=_get_api_key(),
    ).with_structured_output(classifiyQueryOutput)


def _make_evaluator_model():
    """Deterministic - for scoring."""
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.0,
        api_key=_get_api_key(),
    ).with_structured_output(QueryEvaluatorOutput)


def _make_optimizer_model():
    """Powerful - for query improvement."""
    return ChatMistralAI(
        model="mistral-medium-latest",
        temperature=0.5,
        api_key=_get_api_key(),
    ).with_structured_output(QuerySynthesizerOutput)


def _make_main_model():
    """
    Output generation with streaming enabled.
    Real-time streaming from LLM for faster user experience.
    """
    return ChatMistralAI(
        model="mistral-medium-latest",
        timeout=120,
        streaming=True,
        api_key=_get_api_key(),
    )


def _make_decision_model():
    """
    Fast decision-making model for tool selection.
    """
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.3,
        api_key=_get_api_key(),
    )


def query_classifier_node(state: ResearchGraphState) -> str:
    model = _classify_query_model()
    prompt = f"""
    check whether the {state['user_query']} is a casual converation or not.
    """
    response = model.invoke(prompt).result

    return {"classification": response}


def query_synthesizer_node(state: ResearchGraphState) -> dict:
    """Expands user query into diverse, high-quality research sub-queries.
    If document context is available, generates 10 targeted queries.
    Otherwise, generates 30 comprehensive queries for web search.
    """
    context = state.get("context", "")
    has_context = context and context.strip() and len(context) > 10
    
    if has_context:
        logger.info("[QuerySynthesizer] START - Document context available, generating 10 targeted queries")
    else:
        logger.info("[QuerySynthesizer] START - No document context, generating 30 comprehensive queries")
    _flush()

    prompt = query_synthesizer_prompt(state)
    start = time.time()
    model = _make_synthesizer_model()

    try:
        result = model.invoke(prompt)
        queries = result.refined_queries if result.refined_queries else []
    except Exception as e:
        logger.error(f"[QuerySynthesizer] Structured output failed: {e}", exc_info=True)
        _flush()
        queries = []

    elapsed = time.time() - start

    if not queries:
        queries = [state["user_query"]]
        logger.warning(
            "[QuerySynthesizer] Empty result — falling back to original query"
        )

    expected_count = 10 if has_context else 30
    logger.info(f"[QuerySynthesizer] {elapsed:.2f}s | {len(queries)} queries generated (expected: {expected_count})")
    
    # Log a few sample queries
    for i in range(min(3, len(queries))):
        logger.info(f"  Sample {i+1}: {queries[i][:80]}{'...' if len(queries[i]) > 80 else ''}")
    _flush()

    return {"refined_queries": queries}


def query_evaluator_node(state: ResearchGraphState) -> dict:
    """
    Scores each refined query individually (0-10).
    Returns dict mapping query -> score.
    """
    logger.info("[QueryEvaluator] START - Evaluating each query individually")
    _flush()

    prompt = query_evaluator_prompt(state)
    start = time.time()
    model = _make_evaluator_model()

    try:
        result = model.invoke(prompt)
        query_scores = result.query_score if result.query_score is not None else {}
    except Exception as e:
        logger.error(f"[QueryEvaluator] Structured output failed: {e}", exc_info=True)
        _flush()
        query_scores = {q: 10 for q in state.get("refined_queries", [])}

    elapsed = time.time() - start

    if query_scores:
        avg_score = sum(query_scores.values()) / len(query_scores)
        min_score = min(query_scores.values())
        max_score = max(query_scores.values())
        logger.info(
            f"[QueryEvaluator] {elapsed:.2f}s | {len(query_scores)} queries evaluated"
        )
        logger.info(
            f"[QueryEvaluator] Scores - Avg: {avg_score:.1f}, Min: {min_score}, Max: {max_score}"
        )
    else:
        logger.warning("[QueryEvaluator] No scores returned - using fallback")

    _flush()

    return {"query_score": query_scores}


def check_query_quality(state: ResearchGraphState) -> str:
    """
    Routes after evaluation node.
    If ANY query has score < 9, triggers optimizer.
    """
    query_scores = state.get("query_score", {})

    if not query_scores:
        logger.warning(
            "[CheckQuality] No scores found - defaulting to Need_Improvement"
        )
        _flush()
        return "Need_Improvement"

    low_scoring_queries = {q: s for q, s in query_scores.items() if s < 9}

    if low_scoring_queries:
        decision = "Need_Improvement"
        logger.info(
            f"[CheckQuality] {len(low_scoring_queries)}/{len(query_scores)} queries need optimization (score < 9)"
        )
        logger.info(
            f"[CheckQuality] Low-scoring queries: {list(low_scoring_queries.keys())[:3]}..."
        )
    else:
        decision = "Approved"
        logger.info(
            f"[CheckQuality] All {len(query_scores)} queries approved (all scores ≥ 9)"
        )

    _flush()
    return decision


def query_optimizer_node(state: ResearchGraphState) -> dict:
    """
    Improved only weak queries.
    """
    logger.info(
        "[QueryOptimizer] START - Selective optimization (targeting 30 total queries)"
    )
    _flush()

    query_scores = state.get("query_score", {})
    original_queries = state.get("refined_queries", [])

    # Separate queries by score
    queries_to_optimize = []
    queries_to_keep = []

    for query in original_queries:
        score = query_scores.get(query, 0)
        if score < 9:
            queries_to_optimize.append(query)
        else:
            queries_to_keep.append(query)

    logger.info(
        f"[QueryOptimizer] Keeping {len(queries_to_keep)} high-scoring queries (≥9)"
    )
    logger.info(
        f"[QueryOptimizer] Optimizing {len(queries_to_optimize)} low-scoring queries (<9)"
    )
    _flush()

    if not queries_to_optimize:
        logger.info("[QueryOptimizer] No queries need optimization - skipping")
        _flush()
        return {"refined_queries": original_queries}

    prompt = query_optimizer_prompt(state, queries_to_optimize, query_scores)
    start = time.time()
    model = _make_optimizer_model()

    try:
        result = model.invoke(prompt)
        optimized_queries = result.refined_queries if result.refined_queries else []
    except Exception as e:
        logger.error(f"[QueryOptimizer] Structured output failed: {e}", exc_info=True)
        _flush()
        optimized_queries = []

    elapsed = time.time() - start

    if not optimized_queries:
        logger.warning(
            "[QueryOptimizer] Empty result - keeping original low-scoring queries"
        )
        optimized_queries = queries_to_optimize

    final_queries = queries_to_keep + optimized_queries

    logger.info(
        f"[QueryOptimizer] {elapsed:.2f}s | Final: {len(final_queries)} queries"
    )
    logger.info(
        f"[QueryOptimizer] Breakdown: {len(queries_to_keep)} kept + {len(optimized_queries)} optimized"
    )

    if len(final_queries) < 25:
        logger.warning(
            f"[QueryOptimizer] Only {len(final_queries)} queries generated (target: 30)"
        )

    _flush()

    return {"refined_queries": final_queries}


def _extract_json_from_response(text: str) -> dict | None:
    """
    Robustly extract a JSON object from an LLM response.
    Handles raw JSON, markdown code fences, and inline JSON.
    """
    if not text:
        return None

    # 1. Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find first {...} block
    brace_match = re.search(r"\{[\s\S]*?\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def tool_decision_node(state: ResearchGraphState) -> dict:
    queries = state.get("refined_queries", [])
    base_query = state.get("user_query", "")
    input_type = state.get("input_type", "general")

    logger.info("[ToolDecision] START")
    logger.info(f"[ToolDecision] input_type={input_type} | queries={len(queries)}")
    _flush()

    # Different prompts based on whether document is available
    if input_type in ("pdf", "docs"):
        prompt = f"""You are an intelligent research agent with access to uploaded documents AND web search.

Tools Available:
1. document        - Retrieves information from the uploaded document(s)
2. general_search  - Fast fact-checking and instant web responses
3. advanced_search - Deep research, latest news, and current information from the web

Strategy:
- ALWAYS check the document FIRST for relevant information
- If the document is insufficient or outdated, use web search tools
- Select ONLY the most useful queries (max 8)

Base Query: {base_query}

Candidate Queries: {queries}

IMPORTANT: Respond with ONLY a valid JSON object, no markdown, no extra text:
{{"use_tool": true, "tool": "document", "selected_queries": ["q1", "q2"]}}

Allowed values for "tool": "document", "general_search", "advanced_search", "none"
"""
    else:
        prompt = f"""You are an intelligent research agent deciding which tool to use.

Tools:
1. general_search  - Use for quick facts, definitions, general knowledge, recent events (fast)
2. advanced_search - Use for deep research, latest news, post-2024 data, citations needed (thorough)
3. none            - Use only if the query can be answered from pre-trained knowledge alone

Decision Rules:
- Use "advanced_search" when: latest news, current events, real-time data, citations needed, deep research
- Use "general_search" when: simple facts, quick lookups, general knowledge
- Use "none" when: simple math, greetings, or clearly answerable without web search
- Select ONLY the most useful queries (max 8)

Base Query: {base_query}

Candidate Queries: {queries}

IMPORTANT: Respond with ONLY a valid JSON object, no markdown, no extra text:
{{"use_tool": true, "tool": "advanced_search", "selected_queries": ["q1", "q2"]}}

Allowed values for "tool": "general_search", "advanced_search", "none"
"""

    # Use the non-streaming decision model (deterministic, structured output)
    model = _make_decision_model()
    response = model.invoke(prompt)

    raw_content = response.content if hasattr(response, "content") else str(response)
    logger.info(f"[ToolDecision] Raw LLM response: {raw_content[:300]}")
    _flush()

    decision = _extract_json_from_response(raw_content)

    if decision is None:
        # Fallback: use document tool if document available, otherwise advanced_search
        fallback_tool = "document" if input_type in ("pdf", "docs") else "advanced_search"
        logger.warning(
            f"[ToolDecision] JSON parse failed — using fallback tool: {fallback_tool}"
        )
        decision = {
            "use_tool": True,
            "tool": fallback_tool,
            "selected_queries": queries[:8],
        }
    else:
        # Validate tool value is one of the known tools
        valid_tools = {"document", "general_search", "advanced_search", "none"}
        tool_val = decision.get("tool", "none")
        if tool_val not in valid_tools:
            logger.warning(
                f"[ToolDecision] Unknown tool '{tool_val}' in response — correcting to advanced_search"
            )
            decision["tool"] = "advanced_search" if input_type not in ("pdf", "docs") else "document"

    logger.info(f"[ToolDecision] Final decision: use_tool={decision.get('use_tool')} | tool={decision.get('tool')} | queries={len(decision.get('selected_queries', []))}")
    _flush()

    return {"tool_decision": decision}


def make_tool_executor_node(toolkit):
    """Returns a graph node function with toolkit captured in closure.
    Enhanced to support multi-tool execution for document scenarios.
    """

    def tool_executor_node(state: ResearchGraphState) -> dict:
        logger.info("[ToolExecutor] START")
        _flush()

        decision = state.get("tool_decision", {})
        use_tool = decision.get("use_tool", False)
        tool_type = decision.get("tool", "none")
        queries = decision.get("selected_queries", [])

        if not use_tool or tool_type == "none" or not queries:
            logger.info("[ToolExecutor] SKIPPED (No tool needed)")
            _flush()
            return {"tool_results": []}

        tools_list = toolkit.get_tools()
        tools_map = {t.name: t for t in tools_list}

        # Map tool types to actual tools
        tool_mapping = {
            "document": ("document_retrieval_tool", "DocumentRetrieval"),
            "advanced_search": ("Advance_Search_mode", "ExaSearch"),
            "general_search": ("general_search_mode", "DuckDuckGo"),
        }

        if tool_type not in tool_mapping:
            logger.warning(f"[ToolExecutor] Unknown tool type: {tool_type}")
            _flush()
            return {"tool_results": []}

        tool_key, tool_name = tool_mapping[tool_type]
        selected_tool = tools_map.get(tool_key)

        if not selected_tool:
            logger.error(f"[ToolExecutor] Tool not found: {tool_type}")
            _flush()
            return {"tool_results": []}

        logger.info(f"[ToolExecutor] Tool={tool_name} | Queries={len(queries)}")
        _flush()

        tool_results = []
        
        # Execute document retrieval (batch mode)
        if tool_type == "document":
            try:
                start = time.time()
                # Document tool expects dict with 'queries' key
                result = selected_tool.invoke({"queries": queries[:5]})
                elapsed = time.time() - start

                tool_results.append(
                    {
                        "query": f"Batch of {min(len(queries), 5)} queries",
                        "result": result,
                        "tool": tool_name,
                    }
                )

                logger.info(
                    f"[{tool_name}] {elapsed:.2f}s | batch={min(len(queries), 5)}"
                )
                _flush()

                # Check if document results are insufficient
                # If result indicates no relevant info found, try general search as fallback
                if "No relevant information found" in result or len(result.strip()) < 50:
                    logger.info(
                        "[ToolExecutor] Document retrieval insufficient, attempting general search fallback"
                    )
                    _flush()
                    
                    # Try general search for a few queries
                    general_tool = tools_map.get("general_search_mode")
                    if general_tool:
                        for i, query in enumerate(queries[:3], 1):  # Limit fallback to 3 queries
                            try:
                                start = time.time()
                                # General search tool expects dict with 'query' key (singular)
                                web_result = general_tool.invoke({"query": query})
                                elapsed = time.time() - start

                                tool_results.append(
                                    {
                                        "query": query,
                                        "result": web_result,
                                        "tool": "DuckDuckGo (Fallback)",
                                    }
                                )

                                preview = query[:60] + "..." if len(query) > 60 else query
                                logger.info(
                                    f"[DuckDuckGo-Fallback] ({i}/3) '{preview}' | {elapsed:.2f}s"
                                )
                                _flush()

                            except Exception as e:
                                logger.error(
                                    f"[DuckDuckGo-Fallback] ERROR: {e}",
                                    exc_info=True,
                                )
                                _flush()

            except Exception as e:
                logger.error(f"[{tool_name}] ERROR: {e}", exc_info=True)
                _flush()
                tool_results.append(
                    {
                        "query": str(queries[:5]),
                        "result": f"Retrieval error: {str(e)}",
                        "tool": tool_name,
                    }
                )

        else:
            for i, query in enumerate(queries, 1):
                try:
                    start = time.time()
                    result = selected_tool.invoke({"query": query})
                    elapsed = time.time() - start

                    tool_results.append(
                        {
                            "query": query,
                            "result": result,
                            "tool": tool_name,
                        }
                    )

                    preview = query[:60] + "..." if len(query) > 60 else query
                    logger.info(
                        f"[{tool_name}] ({i}/{len(queries)}) '{preview}' | {elapsed:.2f}s"
                    )
                    _flush()

                except Exception as e:
                    preview = query[:40] + "..." if len(query) > 40 else query
                    logger.error(
                        f"[{tool_name}] ERROR on query '{preview}': {e}",
                        exc_info=True,
                    )
                    _flush()

                    tool_results.append(
                        {
                            "query": query,
                            "result": f"Search error: {str(e)}",
                            "tool": tool_name,
                        }
                    )

        logger.info(f"[ToolExecutor] DONE | {len(tool_results)} results collected")
        _flush()

        return {"tool_results": tool_results}

    return tool_executor_node


def make_generate_output_node(agent_role: str):
    """Returns a graph node with agent_role captured in closure."""

    is_research = agent_role.lower() == "research"

    def generate_output_node(state: ResearchGraphState) -> dict:
        logger.info("[GenerateOutput] START")
        _flush()

        tool_results = state.get("tool_results", [])
        queries = state.get("refined_queries") or [state["user_query"]]

        logger.info(
            f"[GenerateOutput] tool_results={len(tool_results)} | queries={len(queries)}"
        )
        _flush()

        # ── Build context string ──
        context_blocks = []
        citation_list = []  # Accumulate URL citations from Exa results
        citation_index = 1

        for i, tr in enumerate(tool_results, 1):
            raw_result = tr["result"]
            tool_name = tr["tool"]

            # If the result is a list of Exa dicts (from advanced search), format with citations
            if isinstance(raw_result, list):
                formatted_sources = []
                for item in raw_result:
                    title = item.get("T", "")
                    url = item.get("U", "")
                    date = item.get("D", "")
                    content = item.get("C", "")
                    ref_label = f"[{citation_index}]"
                    formatted_sources.append(
                        f"{ref_label} **{title}** ({date})\n{content}"
                    )
                    citation_list.append(f"{ref_label} {title} — {url}")
                    citation_index += 1
                result_str = "\n\n".join(formatted_sources)
            else:
                result_str = str(raw_result)

            context_blocks.append(
                f"[Source {i}]\n"
                f"Query : {tr['query']}\n"
                f"Tool  : {tool_name}\n"
                f"Result:\n{result_str}"
            )

        context_str = (
            "\n\n---\n\n".join(context_blocks)
            if context_blocks
            else "No external sources retrieved."
        )

        # Append citation reference list at end of context if URLs found
        if citation_list:
            citation_block = "\n\n---\n\n### Source URLs:\n" + "\n".join(citation_list)
            context_str += citation_block

        if is_research:
            system_prompt = """

                IMPORTANT : 
                if the user is having causal convertation then reply casually.

                You are a deep research assistant providing PhD-level analysis.
                Your output must be comprehensive, technically precise, and organized
                by topic/concept (NOT by query). Cover mechanisms, theory, applications
                limitations, and recent trends. Support claims with the provided sources.
                Use inline citations like [1], [2], etc. where appropriate, matching
                the numbered references in the Source URLs section of the context.
                At the end of your response, include a **References** section listing
                each cited URL in full format: [N] Title — URL"""

        else:
            system_prompt = """
            
                IMPORTANT : 
                if the user is having causal convertation then reply casually.

                You are a knowledgeable research assistant.
                Your output must be clear, structured, and informative. 
                Use headers and bullet points for readability. 
                Be accurate and support answers with the provided context.
                Use inline citations like [1], [2], etc. where source URLs are available.
                At the end, include a **References** section for any cited URLs."""

        user_prompt = f"""

                ### Original Question:
                {state["user_query"]}

                ### Research Angles Explored:
                {chr(10).join(f"- {q}" for q in queries)}

                ### Retrieved Context:
                {context_str}

                ### Instructions:
                - Analyse all research angles first, then synthesize a unified response
                - Provide in-depth information covering each angle
                - Organize your output by topic/concept — NOT by query order
                - Do NOT list or number the sub-queries in your output
                - Use inline citations [N] matching the numbered source URLs when context supports a claim
                - Be thorough but avoid unnecessary repetition
                - The output is for PhD-level readers — be precise and complete
                - End your response with a **References** section listing all cited sources with their URLs
                """

        logger.info("[GenerateOutput] Invoking main LLM...")
        _flush()

        start = time.time()
        model = _make_main_model()

        try:

            output_text = ""
            for chunk in model.stream(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            ):
                if hasattr(chunk, "content") and chunk.content:
                    output_text += chunk.content

            # Fallback if streaming produces no content
            if not output_text:
                response = model.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                output_text = response.content if response.content else ""

        except Exception as e:
            logger.error(f"[GenerateOutput] LLM invocation failed: {e}", exc_info=True)
            _flush()
            output_text = f"Error generating response: {str(e)}"

        elapsed = time.time() - start

        logger.info(
            f"[GenerateOutput] LLM done | {elapsed:.2f}s | {len(output_text)} chars"
        )
        _flush()

        return {"output": output_text}

    return generate_output_node
