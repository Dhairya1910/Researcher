from datetime import datetime


def query_synthesizer_prompt(state):
    context = state.get('context', '')
    
    # If document context is available, generate fewer, more targeted queries
    if context and context.strip():
        return f"""
        You are an expert Query Optimization Agent working with uploaded document content.
        Your task is to generate EXACTLY 10 highly targeted, document-specific sub-queries that will retrieve the most relevant information from the provided context.
        
        ### Context from Document (Summary):
        {context[:2000]}...
        
        
        ### Objectives:
        1. Analyze the user query AND the document context
        2. Generate queries that:
           - Target specific sections/topics mentioned in the document
           - Use terminology and concepts present in the document
           - Extract detailed information relevant to user's question
           - Cover different aspects of the query that the document addresses
        3. Make queries specific enough to retrieve precise chunks from the document
        
        ### Rules:
        - Generate EXACTLY 10 queries (targeted for document context)
        - Queries MUST be tailored to extract info from the provided document
        - Use keywords and phrases that appear in the document context
        - Focus on what the document can answer about the user's query
        - Do NOT ask about information clearly missing from the context
        - Keep queries concise but specific
        
        ### Example:
        User Query: "Explain neural networks"
        Context: "...discusses backpropagation, gradient descent, activation functions..."
        Output:
        [
        "What is backpropagation as described in the document?",
        "How does gradient descent work according to the document?",
        "What activation functions are mentioned in the document?",
        ...
        ]
        
        ### Now process:
        User Query: {state['user_query']}
        Current datetime: {datetime.now()}
        
        **Generate EXACTLY 10 document-targeted queries.**
    """
    

    else:
        return f"""
        You are an expert Query Optimization Agent.
        Your task is to transform a single user query into EXACTLY 30 diverse, high-quality, and information-rich sub-queries that cover the topic comprehensively.
        
        ### Objectives:
        1. Break down the original query into multiple meaningful angles.
        2. Ensure coverage of:
        - Definitions / fundamentals
        - Core concepts & working
        - Architecture / structure (if applicable)
        - Mathematical/theoretical foundations
        - Applications / use cases
        - Advantages & limitations
        - Comparisons (if relevant)
        - Recent developments / trends
        - Practical implementation or examples
        - Performance considerations
        - Best practices
        - Common pitfalls
        - Industry use cases
        - Future directions
        
        3. Generate queries that are:
        - Clear and specific
        - Non-redundant
        - Useful for retrieval (search, RAG, or research agents)
        - Research-oriented (encourage deep exploration)
        
        ### Rules:
        - Do NOT answer the query.
        - Only generate optimized sub-queries.
        - Avoid repeating similar queries with minor wording changes.
        - Adapt the expansion based on the domain (technical, business, general knowledge, etc.).
        - Keep queries concise but meaningful.
        - Cover the topic from beginner to advanced perspectives.
        
        ### Example:
        User Query: "What is CNN?"
        Output:
        [
        "What is a Convolutional Neural Network (CNN)?",
        "How does a CNN work step by step?",
        "What are the main layers in CNN architecture?",
        "How does backpropagation work in CNNs?",
        "What are common applications of CNNs in computer vision?",
        "Advantages and limitations of CNN models",
        "Difference between CNN and traditional neural networks",
        "Recent advancements in CNN architectures",
        "What is the mathematical intuition behind convolution operations?",
        "How do pooling layers work in CNNs?",
        ...
        ]
        
        ### Now process the following query:
        User Query: {state['user_query']}
        Current datetime: {datetime.now()}
        
        **Generate EXACTLY 30 diverse, high-quality sub-queries.**
    """


def query_evaluator_prompt(state):
    return f"""
        You are an expert Query Evaluation Agent.
        Your task is to evaluate the quality of EACH refined query individually and assign a score to each one.

        ### Inputs:
        1. Base Query: The original user query.
        2. Refined Queries: A list of expanded queries.

        ### Evaluation Objective:
        Assess whether EACH refined query effectively contributes to expanding the base query into a diverse, deep, and research-oriented exploration.

        ### Evaluation Criteria (Apply to EACH query individually):
        1. Coverage Depth (0–2)
        - Does THIS query explore the topic in depth?
        - Does it go beyond surface-level understanding?

        2. Topical Diversity (0–2)
        - Does THIS query cover a unique perspective not covered by others?
        - Avoid narrow or repetitive focus.

        3. Relevance (0–2)
        - Is THIS query directly related to the base query?
        - No off-topic or loosely related queries.

        4. Non-Redundancy (0–2)
        - Is THIS query distinct and not just a reworded duplicate of another?

        5. Research Orientation (0–2)
        - Does THIS query encourage deeper exploration, analysis, or learning?
        - (e.g., "how", "why", "comparison", "limitations", "real-world use")
        
        6. For your knowledge Current datetime: {datetime.now()}
        
        ### Scoring:
        - Evaluate EACH query separately.
        - For each query, sum scores from all 5 criteria (Max: 10)
        - Return a dictionary mapping each query to its individual score

        ### Output Format (STRICT):
        {{
            "query1 text here": 8,
            "query2 text here": 6,
            "query3 text here": 9,
            ...
        }}

        ### Rules:
        - Be strict and analytical, not lenient.
        - Penalize redundancy, shallow queries, and lack of diversity.
        - Reward depth, variety, and research-oriented phrasing.
        - Do NOT answer the queries themselves.
        - MUST return a score for EVERY query in the refined_queries list.
        - Scores must be integers between 0 and 10.
        - Expect 10 queries for document mode, 30 queries for research mode.

        ### Input:
        Base Query: {state["user_query"]}  
        Refined Queries ({len(state.get("refined_queries", []))} queries): {state["refined_queries"]}
        
        ### Example Output Format:
        {{
            "What is a CNN?": 5,
            "How does CNN architecture work in detail?": 9,
            "What are the mathematical foundations of convolution?": 10,
            "CNN uses": 3
        }}
    """


def query_optimizer_prompt(state, queries_to_optimize=None, query_scores=None):
    """
    Optimizer prompt that only optimizes low-scoring queries.
    Maintains query count to preserve the 30-query target.
    
    Args:
        state: Graph state with user_query, refined_queries, query_score
        queries_to_optimize: Optional list of specific queries to optimize (if None, uses all)
        query_scores: Optional dict of query->score mappings
    """
    # Use provided queries or fall back to all refined queries
    if queries_to_optimize is None:
        queries_to_optimize = state.get('refined_queries', [])
    
    # Use provided scores or fall back to state scores
    if query_scores is None:
        query_scores = state.get('query_score', {})
    
    # Filter scores to only include queries being optimized
    relevant_scores = {q: query_scores.get(q, 0) for q in queries_to_optimize}
    
    return f"""
        You are an expert Query Refinement Agent.
        Your task is to take queries with LOW SCORES (<9) and improve them into highly detailed, research-oriented, and insight-rich queries.
        For your knowledge Current datatime : {datetime.now()}
        
        ### Inputs:
        1. Base Query: The original user query.
        2. Low-Scoring Queries: Queries that need improvement (score < 9).
        3. Query Scores: Individual scores for each low-scoring query.

        ### Objective:
        Transform ONLY the provided low-scoring queries into superior queries that:
        - Force deep research and exploration
        - Extract high-quality, expert-level insights
        - Cover the topic comprehensively from multiple dimensions
        - Achieve a score of 9 or 10 on the evaluation criteria

        ### Refinement Strategy:

        For each low-scoring query, enhance it by:
        - Adding depth (e.g., "step-by-step", "mathematical intuition", "real-world trade-offs")
        - Making it more specific and less generic
        - Encouraging analytical thinking (e.g., "why", "how", "when", "trade-offs", "limitations")
        - Including practical and implementation perspectives where relevant
        - Expanding vague queries into precise, research-driven ones

        ### Ensure Coverage Across:
        - Fundamental concepts
        - Internal working / mechanisms
        - Architecture / structure (if applicable)
        - Mathematical or theoretical foundations (if applicable)
        - Real-world applications & case studies
        - Advantages, limitations, and trade-offs
        - Comparisons with alternatives
        - Performance considerations
        - Recent advancements / trends

        ### Rules:
        - Do NOT answer the queries.
        - Do NOT simply rephrase — significantly improve depth and clarity.
        - Avoid redundancy — each query must add unique value.
        - Keep queries concise but information-dense.
        - Ensure all queries are directly relevant to the base query.
        - Remove weak, vague, or duplicate queries.
        - **CRITICAL: Output MUST have EXACTLY {len(queries_to_optimize)} queries** (same as input count).
        - Do NOT add extra queries or reduce count - maintain parity with input.

        ### Example:
        Base Query: "What is CNN?"

        Low-Scoring Queries Input (with scores):
        "What is CNN?" (Score: 5)
        "How does CNN work?" (Score: 6)
        "Applications of CNN" (Score: 7)

        Output (EXACTLY 3 queries - same as input):
        [
        "What is a Convolutional Neural Network (CNN) and what core problem does it solve in deep learning?",
        "Explain the step-by-step working of CNN including convolution, activation, pooling, and fully connected layers with mathematical intuition",
        "What are real-world applications of CNNs across industries like healthcare, autonomous driving, and NLP, including specific case studies and performance benchmarks?"
        ]
        
        ### Now process the following input:
        Base Query: {state["user_query"]}  
        Low-Scoring Queries to Optimize: {queries_to_optimize}
        Query Scores: {relevant_scores}
        
        **REMEMBER: Return EXACTLY {len(queries_to_optimize)} optimized queries - no more, no less.**
            """


def generate_output_prompt(state):
    return f"""
        you are a deep research agent your job is to provide user with a deep insight from the QUERIES you are provided with.

            OBJECTIVE : 
            - first analyse all the queries and then answer.
            - provide in-depth information about each and every query. 
            - the output is for PHD student and research so make sure you make no mistakes
            - generate a very detailed output, try to cover each and every query.
            - do not output the queries to user. ie your output should be topic wise

            QUERIES : 
            {state['refined_queries']}
        """
