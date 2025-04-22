from typing import TypedDict, List, Dict, Optional, Annotated, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
import time
import os
import logging
import json
import re
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('search_agent')

def log_step(step_name, message, data=None):
    """Log a step in the search process with optional data."""
    logger.info(f"[{step_name}] {message}")
    if data and isinstance(data, dict):
        try:
            # Pretty print the data with indentation for readability
            formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
            for line in formatted_data.split('\n'):
                logger.info(f"  {line}")
        except:
            logger.info(f"  {data}")
    elif data and isinstance(data, list):
        for item in data:
            logger.info(f"  - {item}")
    elif data:
        logger.info(f"  {data}")

# Define our state
class SearchState(TypedDict):
    # Original query and assessment
    query: str
    query_type: Optional[str]  # "direct", "simple_search", "deep_research"

    # Search-related data
    search_results: List[Dict]
    search_queries: List[str]

    # Research aggregation (for deep_research)
    research_data: Dict

    # Processing status
    verification_status: Optional[bool]
    verification_feedback: Optional[str]
    confidence_score: Optional[float]  # Added confidence score
    attempts: int
    max_attempts: int

    # Response generation
    final_answer: Optional[str]

    # Source tracking for attribution
    sources: List[Dict]  # Added sources tracking

    # Conversation tracking
    messages: Annotated[List[Any], add_messages]

# Initialize LLM and search tool
llm = ChatOpenAI(temperature=0.2)
search_tool = GoogleSerperAPIWrapper()

# Define nodes
def query_assessment(state: SearchState) -> Dict:
    """Assess the query to determine the appropriate search strategy."""
    query = state["query"]

    log_step("QUERY_ASSESSMENT", f"Assessing query: '{query}'")

    prompt = f"""
    Analyze the following query and determine the most appropriate search strategy:

    Query: {query}

    Determine if this query:
    1. Can be answered directly from your knowledge (DIRECT)
    2. Requires a simple search for real-time or specific information (SIMPLE_SEARCH)
    3. Requires deep research with multiple searches and information aggregation (DEEP_RESEARCH)

    Respond only with: DIRECT, SIMPLE_SEARCH, or DEEP_RESEARCH
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    query_type = response.content.strip()

    log_step("QUERY_ASSESSMENT", f"Query classified as: {query_type}")

    # Return updates to state
    return {
        "query_type": query_type,
        "attempts": 0,
        "max_attempts": 3,  # Default max attempts
        "messages": [
            {"role": "system", "content": "Query assessment complete."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    }

def direct_answer(state: SearchState) -> Dict:
    """Generate a direct answer for queries within LLM knowledge."""
    query = state["query"]

    prompt = f"""
    Please answer the following query using your knowledge:

    Query: {query}

    Provide a comprehensive and accurate answer based on your knowledge.
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    # Return updates to state
    return {
        "final_answer": response.content,
        "messages": [
            {"role": "system", "content": "Generating direct answer."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    }

def generate_search_query(state: SearchState) -> Dict:
    """Generate an appropriate search query based on the original query."""
    query = state["query"]
    query_type = state["query_type"]
    attempts = state["attempts"]

    log_step("GENERATE_SEARCH", f"Iteration {attempts + 1}: Generating search query for '{query}'")

    # If we've already generated search queries, refine them
    previous_queries = state.get("search_queries", [])
    previous_results = state.get("search_results", [])

    prompt = f"""
    Based on the original query: "{query}"

    {"And considering previous search attempts:" if previous_queries else ""}
    {"; ".join(previous_queries) if previous_queries else ""}

    {"Previous searches didn't yield satisfactory results." if attempts > 0 else ""}

    Generate an effective search query that will help find the information needed.
    If this is a refinement, try a different approach than previous attempts.

    Respond with only the search query, no explanation.
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    search_query = response.content.strip()

    # Add to the list of search queries
    search_queries = previous_queries + [search_query]

    log_step("GENERATE_SEARCH", f"Generated search query: '{search_query}'")

    # Return updates to state
    return {
        "search_queries": search_queries,
        "messages": [
            {"role": "system", "content": f"Generating search query (attempt {attempts + 1})."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    }

def execute_search(state: SearchState) -> Dict:
    """Execute a search operation using the generated search query."""
    search_queries = state["search_queries"]
    latest_query = search_queries[-1]

    log_step("EXECUTE_SEARCH", f"Searching for: '{latest_query}'")

    try:
        # Execute search
        search_results = search_tool.results(latest_query)

        # Extract searched sites for logging and source tracking
        sites = []
        sources = state.get("sources", [])

        if isinstance(search_results, dict) and "organic" in search_results:
            for i, result in enumerate(search_results["organic"]):
                if "link" in result:
                    site_info = {
                        "url": result["link"],
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "query": latest_query,
                        "position": i + 1
                    }
                    sites.append(result["link"])
                    sources.append(site_info)

        log_step("EXECUTE_SEARCH", f"Search complete. Found {len(sites)} sites:", sites)

        # Add to existing results
        all_results = state.get("search_results", [])
        all_results.append({"query": latest_query, "results": search_results})

        # Return updates to state
        return {
            "search_results": all_results,
            "sources": sources,
            "messages": [
                {"role": "system", "content": f"Search executed for: {latest_query}"}
            ]
        }
    except Exception as e:
        log_step("EXECUTE_SEARCH", f"Search error: {str(e)}")

        # Handle search errors
        return {
            "search_results": state.get("search_results", []) + [
                {"query": latest_query, "results": [], "error": str(e)}
            ],
            "messages": [
                {"role": "system", "content": f"Search error: {str(e)}"}
            ]
        }

def execute_parallel_searches(state: SearchState) -> Dict:
    """Execute multiple searches in parallel for deep research."""
    query = state["query"]

    log_step("DEEP_RESEARCH", f"Generating multiple search queries for deep research: '{query}'")

    # Generate multiple search queries for different aspects
    query_generation_prompt = f"""
    For the deep research query: "{query}"

    Generate 3 different search queries that would help gather comprehensive information.
    Make each query focus on a different aspect or perspective of the topic.
    Return only the 3 queries, one per line, no explanations.
    """

    messages = [HumanMessage(content=query_generation_prompt)]
    response = llm.invoke(messages)

    # Extract the search queries
    search_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]

    log_step("DEEP_RESEARCH", f"Generated {len(search_queries)} search queries:", search_queries)

    # If we have existing queries, add them
    existing_queries = state.get("search_queries", [])
    all_queries = existing_queries + search_queries

    # Execute searches sequentially (instead of in parallel)
    search_results = []
    all_sites = []
    sources = state.get("sources", [])

    for query in search_queries:
        log_step("DEEP_RESEARCH", f"Executing search for: '{query}'")
        try:
            results = search_tool.results(query)

            # Extract searched sites for logging and source tracking
            sites = []

            if isinstance(results, dict) and "organic" in results:
                for i, result in enumerate(results["organic"]):
                    if "link" in result:
                        site_info = {
                            "url": result["link"],
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "query": query,
                            "position": i + 1
                        }
                        sites.append(result["link"])
                        all_sites.append(result["link"])
                        sources.append(site_info)

            log_step("DEEP_RESEARCH", f"Search complete for '{query}'. Found {len(sites)} sites:", sites)
            search_results.append({"query": query, "results": results})
        except Exception as e:
            log_step("DEEP_RESEARCH", f"Search error for '{query}': {str(e)}")
            search_results.append({"query": query, "results": [], "error": str(e)})

    log_step("DEEP_RESEARCH", f"All searches complete. Found {len(all_sites)} total sites.")

    # Add to existing results
    existing_results = state.get("search_results", [])
    all_results = existing_results + search_results

    # Return updates to state
    return {
        "search_queries": all_queries,
        "search_results": all_results,
        "sources": sources,
        "messages": [
            {"role": "system", "content": "Executed parallel searches for deep research."},
            {"role": "user", "content": query_generation_prompt},
            {"role": "assistant", "content": response.content}
        ]
    }

def analyze_simple_search(state: SearchState) -> Dict:
    """Analyze search results for simple search queries to find the most relevant information."""
    query = state["query"]
    search_results = state["search_results"]

    if not search_results or not search_results[-1].get("results"):
        # If no results or error, return state with verification failed
        return {
            "verification_status": False,
            "verification_feedback": "No search results found.",
            "messages": [
                {"role": "system", "content": "Search analysis failed: No results found."}
            ]
        }

    latest_results = search_results[-1]["results"]

    prompt = f"""
    Analyze these search results for the query: "{query}"

    Search Results:
    {latest_results}

    Extract the most relevant piece of information that directly answers the query.
    If you cannot find a clear answer, state "INSUFFICIENT_DATA".
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    # Check if the analysis found sufficient data
    if "INSUFFICIENT_DATA" in response.content:
        verification_status = False
        verification_feedback = "Insufficient data found in search results."
        final_answer = None
    else:
        # Generate a potential answer
        answer_prompt = f"""
        Based on the search results, provide a comprehensive answer to:

        Original query: {query}

        Relevant information: {response.content}

        Provide a clear, direct answer with source attribution.
        """

        answer_messages = [HumanMessage(content=answer_prompt)]
        answer_response = llm.invoke(answer_messages)

        verification_status = True  # Will be verified later
        verification_feedback = None
        final_answer = answer_response.content

    # Return updates to state
    return {
        "verification_status": verification_status,
        "verification_feedback": verification_feedback,
        "final_answer": final_answer,
        "attempts": state["attempts"] + 1,
        "messages": [
            {"role": "system", "content": "Analyzing search results."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": answer_prompt if final_answer else ""},
            {"role": "assistant", "content": final_answer if final_answer else ""}
        ]
    }

def aggregate_research_data(state: SearchState) -> Dict:
    """Aggregate search results for deep research queries into a structured format."""
    query = state["query"]
    search_results = state["search_results"]
    existing_research = state.get("research_data", {})

    # Combine all search results for analysis
    all_results = []
    for result_set in search_results:
        if result_set.get("results"):
            all_results.append({
                "query": result_set["query"],
                "results": result_set["results"]
            })

    if not all_results:
        return {
            "messages": [
                {"role": "system", "content": "No valid results to aggregate."}
            ]
        }

    prompt = f"""
    Analyze these search results for the in-depth research query: "{query}"

    Search Results:
    {all_results}

    Current Research Data:
    {existing_research}

    Extract key information, facts, and insights from these search results.
    Organize them into a structured format with:
    1. Key findings
    2. Supporting evidence
    3. Different perspectives or viewpoints
    4. Sources and their credibility

    Focus on adding new information or validating existing information.
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    # Structure the research data
    structure_prompt = f"""
    Based on your analysis, create a structured research data object with the following format:

    ```python
    {{
        "key_findings": [list of main findings],
        "supporting_evidence": {{finding: [evidence list]}},
        "perspectives": [list of different viewpoints],
        "sources": [list of sources with credibility assessment],
        "gaps": [list of information still missing],
        "confidence": 0.0-1.0 (how confident are we in these findings)
    }}
    ```

    Your analysis:
    {response.content}

    Return ONLY the Python dictionary, no other text.
    """

    structure_messages = [HumanMessage(content=structure_prompt)]
    structure_response = llm.invoke(structure_messages)

    # Extract the dictionary
    import re
    dict_match = re.search(r'```python(.*?)```', structure_response.content, re.DOTALL)
    if dict_match:
        dict_text = dict_match.group(1).strip()
    else:
        dict_text = structure_response.content.strip()

    # Try to parse the dictionary
    try:
        import ast
        updated_research = ast.literal_eval(dict_text)
    except:
        # Fallback
        updated_research = {
            "raw_analysis": response.content,
            "previous": existing_research
        }

    # Return updates to state
    return {
        "research_data": updated_research,
        "messages": [
            {"role": "system", "content": "Aggregating research data."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": structure_prompt},
            {"role": "assistant", "content": structure_response.content}
        ]
    }

def evaluate_research_progress(state: SearchState) -> Dict:
    """Evaluate if the research is complete or if more searches are needed."""
    query = state["query"]
    research_data = state.get("research_data", {})
    attempts = state["attempts"]
    max_attempts = state["max_attempts"]

    prompt = f"""
    Evaluate the current research progress for: "{query}"

    Current Research Data:
    {research_data}

    Current attempt: {attempts + 1} of maximum {max_attempts}

    Determine if the research is complete or if more information is needed.
    Consider:
    - Have we answered all aspects of the query?
    - Is the information comprehensive and accurate?
    - Are there any gaps or inconsistencies?

    Respond with either:
    - COMPLETE: If the research has sufficient information
    - INCOMPLETE: If more research is needed, specifying what information is missing
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    # Check if research is complete
    is_complete = "COMPLETE" in response.content

    # Get feedback on what's missing if incomplete
    verification_status = is_complete
    verification_feedback = response.content if not is_complete else None

    # If complete, generate a final answer
    final_answer = None
    new_messages = [
        {"role": "system", "content": "Evaluating research progress."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    if is_complete:
        summary_prompt = f"""
        Based on the completed research, provide a comprehensive answer to: "{query}"

        Research data: {research_data}

        Synthesize all the information into a clear, well-structured response.
        Include key points, supporting evidence, different perspectives, and appropriate nuance.
        Cite sources where applicable.
        """

        summary_messages = [HumanMessage(content=summary_prompt)]
        summary_response = llm.invoke(summary_messages)
        final_answer = summary_response.content

        new_messages.extend([
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": final_answer}
        ])

    # Return updates to state
    return {
        "verification_status": verification_status,
        "verification_feedback": verification_feedback,
        "final_answer": final_answer,
        "attempts": state["attempts"] + 1,
        "messages": new_messages
    }

def verification(state: SearchState) -> Dict:
    """Verify the correctness and completeness of the final answer and assign a confidence score."""
    query = state["query"]
    final_answer = state["final_answer"]
    query_type = state["query_type"]
    attempts = state["attempts"]

    log_step("VERIFICATION", f"Iteration {attempts}: Verifying answer for '{query}'")

    prompt = f"""
    Verify the correctness and completeness of this answer to: "{query}"

    Answer: {final_answer}

    Query type: {query_type}

    Evaluate the answer based on:
    - Accuracy: Is the information correct?
    - Completeness: Does it fully address the query?
    - Relevance: Is it directly addressing what was asked?
    - Clarity: Is it clear and well-presented?

    Assign a confidence score from 0.0 to 1.0, where:
    - 0.0-0.3: Low confidence (insufficient or potentially incorrect information)
    - 0.4-0.7: Medium confidence (generally correct but may have gaps or uncertainties)
    - 0.8-1.0: High confidence (comprehensive and accurate)

    Respond in the following format:
    SCORE: [confidence score between 0.0-1.0]
    STATUS: [VERIFIED or NEEDS_IMPROVEMENT]
    FEEDBACK: [your feedback on the answer, if any]
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    # Parse the response to extract confidence score and verification status
    confidence_score = 0.0
    is_verified = False
    verification_feedback = None

    # Extract confidence score
    score_match = re.search(r'SCORE:\s*(0?\.\d+|[01])', response.content)
    if score_match:
        try:
            confidence_score = float(score_match.group(1))
        except ValueError:
            confidence_score = 0.0

    # Extract verification status
    status_match = re.search(r'STATUS:\s*(VERIFIED|NEEDS_IMPROVEMENT)', response.content)
    if status_match:
        is_verified = status_match.group(1) == "VERIFIED"
    else:
        is_verified = "VERIFIED" in response.content

    # Extract feedback
    feedback_match = re.search(r'FEEDBACK:\s*(.*?)(?:\n|$)', response.content, re.DOTALL)
    if feedback_match:
        verification_feedback = feedback_match.group(1).strip()
    elif not is_verified:
        verification_feedback = response.content

    log_step("VERIFICATION",
             f"Verification {'passed' if is_verified else 'failed'} with confidence {confidence_score:.2f}",
             {"feedback": verification_feedback} if not is_verified else None)

    # Return updates to state
    return {
        "verification_status": is_verified,
        "verification_feedback": verification_feedback,
        "confidence_score": confidence_score,
        "messages": [
            {"role": "system", "content": "Verifying answer."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    }

def summarize_answer(state: SearchState) -> Dict:
    """Generate a final, polished answer based on the verified information with source attribution."""
    query = state["query"]
    final_answer = state["final_answer"]
    query_type = state["query_type"]
    verification_feedback = state.get("verification_feedback")
    confidence_score = state.get("confidence_score", 0.0)
    attempts = state["attempts"]
    sources = state.get("sources", [])

    log_step("FINAL_ANSWER", f"Summarizing final answer after {attempts} iterations with confidence {confidence_score:.2f}")

    # Extract top sources for attribution (limiting to 3-5 most relevant)
    top_sources = []
    if sources:
        # Sort sources by position (assuming lower position = higher relevance)
        sorted_sources = sorted(sources, key=lambda x: x.get("position", 999))
        # Take top 5 unique sources
        unique_urls = set()
        for source in sorted_sources:
            if source["url"] not in unique_urls and len(top_sources) < 5:
                top_sources.append(source)
                unique_urls.add(source["url"])

    if top_sources:
        log_step("FINAL_ANSWER", f"Top sources for attribution: {len(top_sources)}",
                 [f"{s.get('title', 'Untitled')} - {s['url']}" for s in top_sources])

    prompt = f"""
    Refine and summarize the final answer to: "{query}"

    Current answer: {final_answer}

    Query type: {query_type}

    {"Verification feedback: " + verification_feedback if verification_feedback else ""}

    Confidence score: {confidence_score:.2f}

    Sources to attribute (include these in your answer):
    {json.dumps(top_sources, indent=2) if top_sources else "No specific sources to attribute."}

    Create a polished, comprehensive response that:
    - Directly addresses the query
    - Is well-structured and clear
    - Provides all relevant information
    - Maintains an appropriate level of detail
    - Uses a professional and helpful tone
    - Includes appropriate source attribution at the end of the answer

    If the confidence score is less than 0.7, acknowledge any limitations or uncertainties in the answer.
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    log_step("FINAL_ANSWER", "Answer generation complete")

    # Return updates to state
    return {
        "final_answer": response.content,
        "messages": [
            {"role": "system", "content": "Summarizing final answer."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
    }

# Define routing logic
def route_by_query_type(state: SearchState) -> str:
    """Route based on query assessment."""
    query_type = state["query_type"]

    if query_type == "DIRECT":
        return "direct"
    elif query_type == "SIMPLE_SEARCH":
        return "simple_search"
    elif query_type == "DEEP_RESEARCH":
        return "deep_research"
    else:
        # Default to simple search if assessment is unclear
        return "simple_search"

def check_verification(state: SearchState) -> str:
    """Check if verification passed."""
    verification_status = state.get("verification_status", False)
    attempts = state["attempts"]
    max_attempts = state["max_attempts"]

    if verification_status:
        return "verified"
    elif attempts >= max_attempts:
        return "max_attempts_reached"
    else:
        return "retry"

def check_research_progress(state: SearchState) -> str:
    """Check if research is complete or needs more searches."""
    verification_status = state.get("verification_status", False)
    attempts = state["attempts"]
    max_attempts = state["max_attempts"]

    if verification_status:
        return "complete"
    elif attempts >= max_attempts:
        return "max_attempts_reached"
    else:
        return "continue"

# Create the graph
search_graph = StateGraph(SearchState)

# Add nodes
search_graph.add_node("query_assessment", query_assessment)
search_graph.add_node("direct_answer", direct_answer)
search_graph.add_node("generate_search_query", generate_search_query)
search_graph.add_node("execute_search", execute_search)
search_graph.add_node("execute_parallel_searches", execute_parallel_searches)
search_graph.add_node("analyze_simple_search", analyze_simple_search)
search_graph.add_node("aggregate_research_data", aggregate_research_data)
search_graph.add_node("evaluate_research_progress", evaluate_research_progress)
search_graph.add_node("verification", verification)
search_graph.add_node("summarize_answer", summarize_answer)

# Add edges for the workflow
search_graph.add_edge(START, "query_assessment")

# Route based on query type
search_graph.add_conditional_edges(
    "query_assessment",
    route_by_query_type,
    {
        "direct": "direct_answer",
        "simple_search": "generate_search_query",
        "deep_research": "execute_parallel_searches"
    }
)

# Direct answer path
search_graph.add_edge("direct_answer", "verification")

# Simple search path
search_graph.add_edge("generate_search_query", "execute_search")
search_graph.add_edge("execute_search", "analyze_simple_search")
search_graph.add_edge("analyze_simple_search", "verification")

# Deep research path
search_graph.add_edge("execute_parallel_searches", "aggregate_research_data")
search_graph.add_edge("aggregate_research_data", "evaluate_research_progress")

# Research progress check
search_graph.add_conditional_edges(
    "evaluate_research_progress",
    check_research_progress,
    {
        "complete": "verification",
        "continue": "generate_search_query",
        "max_attempts_reached": "verification"
    }
)

# Verification handling
search_graph.add_conditional_edges(
    "verification",
    check_verification,
    {
        "verified": "summarize_answer",
        "retry": "generate_search_query",
        "max_attempts_reached": "summarize_answer"
    }
)

# Final step
search_graph.add_edge("summarize_answer", END)

# Compile the graph
compiled_search_graph = search_graph.compile()

# Usage example
def run_search_agent(query: str):
    """Run the search agent with a given query."""
    print(f"\n{'=' * 80}")
    print(f"SEARCH QUERY: {query}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    initial_state = {
        "query": query,
        "query_type": None,
        "search_results": [],
        "search_queries": [],
        "research_data": {},
        "verification_status": None,
        "verification_feedback": None,
        "confidence_score": None,
        "attempts": 0,
        "max_attempts": 3,
        "final_answer": None,
        "sources": [],
        "messages": []
    }

    result = compiled_search_graph.invoke(initial_state)

    # Post-execution summary
    execution_time = time.time() - start_time
    confidence = result.get('confidence_score', 0.0)

    print(f"\n{'=' * 80}")
    print(f"SEARCH SUMMARY:")
    print(f"{'=' * 80}")
    print(f"Query: {query}")
    print(f"Query type: {result['query_type']}")
    print(f"Total iterations: {result['attempts']}")
    print(f"Confidence score: {confidence:.2f} ({get_confidence_level(confidence)})")
    print(f"Execution time: {execution_time:.2f} seconds")

    # Count and display unique sites searched
    all_sites = set()
    for source in result.get("sources", []):
        if "url" in source:
            all_sites.add(source["url"])

    if all_sites:
        print(f"\nSites searched ({len(all_sites)}):")
        for site in sorted(all_sites):
            print(f"  - {site}")

    print(f"\n{'=' * 80}")
    print("FINAL ANSWER:")
    print(f"{'=' * 80}")

    return result["final_answer"]

def get_confidence_level(score):
    """Return a human-readable confidence level based on the score."""
    if score >= 0.8:
        return "High confidence"
    elif score >= 0.4:
        return "Medium confidence"
    else:
        return "Low confidence"

# Example usage
if __name__ == "__main__":
    queries = [
        "What is the theory of relativity?",
        "Who won the Super Bowl in 2024?",
        "What are the current global approaches to quantum computing research?"
    ]

    for query in queries:
        print(f"\n\nQuery: {query}")
        print("-" * 50)
        answer = run_search_agent(query)
        print(f"Answer: {answer}")
        print("=" * 80)