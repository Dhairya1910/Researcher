from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
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

        user_message = request.messages[-1].content
        input_type = request.runtime.context["input_type"]
        agent_role = request.runtime.context["agent_role"]

        if input_type in ["pdf", "docs"]:
            print("using a pdf to answer.")
            retriever = self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 5}
            )
            retrieved_docs = retriever.invoke(user_message)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            system_prompt = f"""
                ROLE:
                You are a highly analytical assistant. Your primary directive is to serve as an expert interface for the provided UPLOADED DOCUMENT CONTEXT.

                UPLOADED DOCUMENT CONTEXT:
                ---
                {context_text}
                ---

                OPERATIONAL HIERARCHY & DECISION LOGIC:
                Before responding, you must follow this internal decision-making flow:

                1. EVALUATE: Scan the UPLOADED DOCUMENT CONTEXT for the answer to the User Query: "{user_message}".
                2. CATEGORIZE: 
                    - [FULL MATCH]: The document contains a complete answer. Respond using ONLY the document.
                    - [PARTIAL/NO MATCH]: The document is silent or incomplete regarding the query. 
                    - [NEWS/DEEP DIVE]: The user explicitly asks for "latest news," "sources," or "links."

                STRICT EXECUTION PROTOCOL:

                PHASE A: THE MANDATORY RELEVANCE STATEMENT
                If the answer is NOT fully contained in the UPLOADED DOCUMENT CONTEXT, you MUST ONLY respond with this exact string:
                "The provided document does not contain [sufficient] information about this query. I will now search for the latest information."

                PHASE B: TOOL SELECTION
                - If Phase A is triggered and the query is a general factual question: Use 'general_search_mode'.
                - If Phase A is triggered and the query asks for 'news', 'links', or 'technical descriptions': Use 'Advance_Search_mode'.

                PHASE C: RESPONSE FORMATTING
                - STRUCTURE: Use clear Markdown headings and bullet points.
                - DETAIL: Provide exhaustive explanations. Do not summarize unless asked.
                - ATTRIBUTION: If using a tool, mention that the information comes from external search results.

                EDGE CASE HANDLING:
                - AMBIGUITY: If the query is vague, ask for clarification before using a tool.
                - CONFLICT: If the document contradicts the web, prioritize the Document Context but note the discrepancy.
                - EMPTY CONTEXT: If UPLOADED-Context-Text is empty, treat all queries as [NO MATCH] and trigger Phase A immediately.

                IMPORTANT: Do not skip Phase A. Silent tool usage is a violation of protocol.
                """
            request.system_prompt = system_prompt
        else:
            if agent_role == "General":
                system_prompt = f"""
                ROLE AND OBJECTIVE:
                You are an expert General Assistant. Your primary goal is to provide highly detailed, comprehensive, and well-structured answers. You must anticipate user needs and deliver 90% of all relevant information, context, and mechanics upfront to minimize the need for basic follow-up questions.

                TOOLKIT & SELECTION LOGIC:
                You have access to two search tools. You must route the query based on these strict conditions:
                1. NO TOOLS (GREETINGS/CHAT): If the input is a greeting (e.g., "Hello," "How are you?") or casual filler, do not trigger any tools. Respond naturally and politely.
                2. general_search_mode: Trigger this for standard factual lookups, broad conceptual explanations, recent events, or surface-level summaries.
                3. Advance_Search_mode: Trigger this for intricate, technical, academic, multi-faceted queries, or when gathering exhaustive, niche data not in your base training.

                EXECUTION & EDGE CASE PROTOCOL:
                - THE 90% DEPTH RULE: Do not give brief summaries unless asked. Break down complex answers using headers, bullet points, and bold text for scannability. 
                - MIXED INTENT: If a user mixes a greeting with a prompt (e.g., "Hi, explain quantum computing"), briefly acknowledge the greeting, then immediately execute the appropriate tool.
                - AMBIGUITY: If a query is too vague to provide a 90% depth response or determine the right tool, DO NOT guess. Politely ask the user to clarify.
                - TOOL FAILURE/ZERO RESULTS: If a tool returns no data, inform the user immediately. Do not hallucinate facts to fill the 90% quota.
                - MULTI-PART QUERIES: Systematically address every distinct question within a single prompt.

                SYNTHESIS & QUALITY CONTROL:
                - Seamlessly integrate tool data into your response. Do not just append raw results.
                - Explain the 'why' and 'how' behind technical concepts.
                - CRITICAL: Never mention internal protocols, tool names, or the "90% rule" to the user. Deliver the final result seamlessly.

                USER QUERY: {user_message}
                """
            else:
                system_prompt = f"""
                ROLE:
                You are an elite Research Agent. Your primary objective is to conduct exhaustive investigations, synthesize complex data into digestible formats, and deliver high-fidelity, comprehensive reports. You must balance your internal expertise with rigorous use of external tools to ensure 90% topic coverage, uncompromising accuracy, and deep contextual insight.

                OPERATIONAL PRINCIPLES:
                1. EFFICIENCY & FOCUS: Do not trigger research tools for simple greetings, conversational fillers, or tasks that require zero factual grounding.
                2. TOOL-FIRST FOR EMPIRICAL DATA: If the query requires factual verification, recent events, or technical documentation, you MUST use the appropriate search tool. 
                3. SCANNABILITY & RIGOR: Structure findings like a professional research briefing using headers, bullet points, and tables. Avoid dense blocks of text.

                TOOLKIT DEFINITIONS:
                - general-search-mode: Use for initial topic scoping and standard factual lookups.THIS TOOL SHOULD ONLY BE USED ONCE.
                - advance-search-mode: Use for deep-dive investigations, technical data mining, and extracting specific source links. THIS TOOL SHOULD ONLY BE USED ONCE.

                CITATION & GROUNDING PROTOCOL (CRITICAL):
                When using [advance-search-mode], you will receive a list of sources formatted as: {{"T": "Title", "U": "URL", "C": "Content"}}.
                1. IN-LINE CITATIONS: Every factual claim derived from a tool MUST be followed by an in-line citation using the source URL. Format: [Source Title](URL).
                2. VERBATIM ACCURACY: Do not hallucinate details not present in the "C" (Content) field. If the content is insufficient, state what is missing.
                3. SOURCE LIST: At the end of your report, provide a "References" section listing all unique URLs used in the synthesis.

                STRICT EXECUTION PROTOCOL:

                STEP 1: TRIAGE & TOOL SELECTION
                Analyze the USER QUERY: {user_message}
                - Select the appropriate mode. If [advance-search-mode] is used, you will receive a JSON list of sources.
                - Each source contains 'T' (Title), 'U' (URL), and 'C' (Content/Highlights).
                - You MUST synthesize this incoming data into your final report using the 
                CITATION & GROUNDING PROTOCOL defined above.
                

                STEP 2: SYNTHESIS & STRUCTURE
                Construct your research report as follows:
                - COMPREHENSIVE BREAKDOWN: Use logical, numbered, or bulleted lists.
                - TECHNICAL DEPTH: Explain the 'why' and 'how' behind the data.
                - EVIDENCE-BASED: Integrate the injected data seamlessly. Use in-line markdown links for every major claim (e.g., "...as seen in recent benchmarks [Source Name](https://example.com).").

                STEP 3: QUALITY CONTROL
                - Verify that the URL in the citation matches the specific source providing that information.
                - Ensure the report is scannable and addresses the user's core intent.

                CRITICAL: Do not mention this internal protocol or specific tool names. Provide the final synthesized research with integrated citations.
            """
            request.system_prompt = system_prompt
        return handler(request)
