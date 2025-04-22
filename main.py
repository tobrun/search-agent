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
from urllib.parse import urlparse

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

def is_blacklisted(url, blacklist):
    """Check if a URL is from a blacklisted domain."""
    if not blacklist:
        return False

    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    # Handle www prefix and extract base domain
    if domain.startswith('www.'):
        domain = domain[4:]

    # Check for exact domain match or subdomain match
    for blacklisted_domain in blacklist:
        blacklisted_domain = blacklisted_domain.lower()
        # Remove www prefix if present
        if blacklisted_domain.startswith('www.'):
            blacklisted_domain = blacklisted_domain[4:]

        # Check for exact match or if URL is a subdomain of blacklisted domain
        if domain == blacklisted_domain or domain.endswith('.' + blacklisted_domain):
            return True

    return False

# Define our state
class SearchState(TypedDict):
    # Original query and assessment
    query: str
    query_type: Optional[str]  # "direct", "simple_search", "deep_research"

    # Search-related data
    search_results: List[Dict]
    search_queries: List[str]
    blacklist: List[str]  # List of domains to blacklist

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
    blacklist = state.get("blacklist", [])

    log_step("QUERY_ASSESSMENT", f"Assessing query: '{query}'")

    prompt = f"""
    You are a query classification expert with deep knowledge of information retrieval systems.

    TASK: Analyze this query and determine the optimal search strategy.

    QUERY: "{query}"

    CONTEXT FACTORS:
    - Your current knowledge cutoff is October 2024
    - Technical, scientific, or academic queries often benefit from deep research
    - Current events, statistics, or specific facts typically require search verification
    - General knowledge, concepts, or explanations may be answerable directly
    {f"- The following domains are blacklisted: {', '.join(blacklist)}" if blacklist else ""}

    CLASSIFICATION OPTIONS:

    1. DIRECT: Query can be answered comprehensively from your knowledge base without external search.
       Examples: "Explain photosynthesis", "How does inflation work?", "What is the theory of relativity?"

    2. SIMPLE_SEARCH: Query requires verification or specific information that can be found with a single targeted search.
       Examples: "Who won the Super Bowl in 2024?", "What is the population of Tokyo?", "Latest iPhone release date"

    3. DEEP_RESEARCH: Query requires comprehensive research with multiple searches and information synthesis.
       Examples: "Compare current approaches to quantum computing", "What are the economic impacts of climate change policies?", "How has artificial intelligence changed medical diagnostics?"

    Respond ONLY with one of these three classifications: DIRECT, SIMPLE_SEARCH, or DEEP_RESEARCH
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    query_type = response.content.strip()

    # Extract just the classification label if there's additional text
    if "DIRECT" in query_type:
        query_type = "DIRECT"
    elif "SIMPLE_SEARCH" in query_type:
        query_type = "SIMPLE_SEARCH"
    elif "DEEP_RESEARCH" in query_type:
        query_type = "DEEP_RESEARCH"

    log_step("QUERY_ASSESSMENT", f"Query classified as: {query_type}")

    # Determine max attempts based on query type
    max_attempts = 2 if query_type == "DIRECT" else (3 if query_type == "SIMPLE_SEARCH" else 4)

    # Return updates to state
    return {
        "query_type": query_type,
        "attempts": 0,
        "max_attempts": max_attempts,  # Adaptive max attempts based on query type
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
    blacklist = state.get("blacklist", [])

    log_step("EXECUTE_SEARCH", f"Searching for: '{latest_query}'")
    if blacklist:
        log_step("EXECUTE_SEARCH", f"Using domain blacklist: {blacklist}")

    try:
        # Execute search
        search_results = search_tool.results(latest_query)

        # Extract searched sites for logging and source tracking
        sites = []
        filtered_sites = []
        sources = state.get("sources", [])

        if isinstance(search_results, dict) and "organic" in search_results:
            # Filter out blacklisted domains and track what's filtered
            filtered_results = []
            for i, result in enumerate(search_results["organic"]):
                if "link" in result:
                    url = result["link"]
                    if is_blacklisted(url, blacklist):
                        filtered_sites.append(url)
                        continue

                    site_info = {
                        "url": url,
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "query": latest_query,
                        "position": i + 1
                    }
                    sites.append(url)
                    sources.append(site_info)
                    filtered_results.append(result)

            # Replace original results with filtered results
            search_results["organic"] = filtered_results

        if filtered_sites:
            log_step("EXECUTE_SEARCH", f"Filtered {len(filtered_sites)} blacklisted sites:", filtered_sites)

        log_step("EXECUTE_SEARCH", f"Search complete. Found {len(sites)} sites after filtering:", sites)

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
    blacklist = state.get("blacklist", [])

    log_step("DEEP_RESEARCH", f"Generating multiple search queries for deep research: '{query}'")
    if blacklist:
        log_step("DEEP_RESEARCH", f"Using domain blacklist: {blacklist}")

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
    filtered_sites = []
    sources = state.get("sources", [])

    for query in search_queries:
        log_step("DEEP_RESEARCH", f"Executing search for: '{query}'")
        try:
            results = search_tool.results(query)

            # Extract searched sites for logging and source tracking
            sites = []
            query_filtered_sites = []

            if isinstance(results, dict) and "organic" in results:
                # Filter out blacklisted domains
                filtered_results = []
                for i, result in enumerate(results["organic"]):
                    if "link" in result:
                        url = result["link"]
                        if is_blacklisted(url, blacklist):
                            query_filtered_sites.append(url)
                            filtered_sites.append(url)
                            continue

                        site_info = {
                            "url": url,
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "query": query,
                            "position": i + 1
                        }
                        sites.append(url)
                        all_sites.append(url)
                        sources.append(site_info)
                        filtered_results.append(result)

                # Replace original results with filtered results
                results["organic"] = filtered_results

            if query_filtered_sites:
                log_step("DEEP_RESEARCH", f"Filtered {len(query_filtered_sites)} blacklisted sites for '{query}':",
                         query_filtered_sites)

            log_step("DEEP_RESEARCH", f"Search complete for '{query}'. Found {len(sites)} sites after filtering:", sites)
            search_results.append({"query": query, "results": results})
        except Exception as e:
            log_step("DEEP_RESEARCH", f"Search error for '{query}': {str(e)}")
            search_results.append({"query": query, "results": [], "error": str(e)})

    if filtered_sites:
        log_step("DEEP_RESEARCH", f"Total of {len(filtered_sites)} blacklisted sites filtered across all searches")

    log_step("DEEP_RESEARCH", f"All searches complete. Found {len(all_sites)} total sites after filtering.")

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
    sources = state.get("sources", [])

    log_step("VERIFICATION", f"Iteration {attempts}: Verifying answer for '{query}'")

    # Extract source information for verification
    source_info = ""
    if sources:
        top_sources = sources[:5] if len(sources) > 5 else sources
        source_info = "Sources used:\n" + "\n".join([
            f"- {s.get('title', 'Untitled')} ({s['url']})" for s in top_sources
        ])

    prompt = f"""
    You are a fact-checking expert tasked with verifying an answer's accuracy and completeness.

    QUERY: "{query}"

    ANSWER TO VERIFY: {final_answer}

    QUERY TYPE: {query_type}

    {source_info}

    VERIFICATION CRITERIA:
    1. Accuracy (40%): Are all stated facts correct? Are there any errors or misrepresentations?
    2. Completeness (30%): Does the answer fully address all aspects of the query?
    3. Relevance (20%): Is the answer directly addressing what was asked without unnecessary tangents?
    4. Clarity (10%): Is the information presented in a clear, well-structured manner?

    For each criterion, assign a sub-score from 0.0-1.0, then calculate a weighted final score.

    Confidence score ranges:
    - 0.0-0.3: Low confidence (insufficient or potentially incorrect information)
    - 0.4-0.7: Medium confidence (generally correct but may have gaps or uncertainties)
    - 0.8-1.0: High confidence (comprehensive and accurate)

    Respond in the following structured format:
    ACCURACY_SCORE: [score] - [brief justification]
    COMPLETENESS_SCORE: [score] - [brief justification]
    RELEVANCE_SCORE: [score] - [brief justification]
    CLARITY_SCORE: [score] - [brief justification]
    FINAL_SCORE: [weighted average]
    STATUS: [VERIFIED or NEEDS_IMPROVEMENT]
    FEEDBACK: [specific improvements needed or confirmation of quality]
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

    # Determine response style based on query type and confidence
    response_style = "academic" if query_type == "DEEP_RESEARCH" else "concise"
    if confidence_score < 0.4:
        response_style = "cautious"

    prompt = f"""
    You are an expert knowledge synthesizer tasked with creating a final answer for: "{query}"

    CURRENT ANSWER: {final_answer}

    QUERY TYPE: {query_type}

    VERIFICATION FEEDBACK: {verification_feedback if verification_feedback else "No specific feedback provided."}

    CONFIDENCE SCORE: {confidence_score:.2f}

    RESPONSE STYLE: {response_style}

    SOURCES TO ATTRIBUTE:
    {json.dumps(top_sources, indent=2) if top_sources else "No specific sources to attribute."}

    GUIDELINES:

    1. Structure:
       - Begin with a direct answer to the query
       - Follow with supporting details and context
       - End with source attribution

    2. Style Adaptation:
       - For "concise" style: Clear, straightforward language with minimal technical jargon
       - For "academic" style: More detailed, authoritative tone with relevant terminology
       - For "cautious" style: Acknowledge limitations clearly, use qualifying language

    3. Confidence Handling:
       - High confidence (>0.7): Present information with certainty
       - Medium confidence (0.4-0.7): Acknowledge some limitations or areas of uncertainty
       - Low confidence (<0.4): Clearly state limitations and present information as possibilities

    4. Source Attribution:
       - Include a "Sources:" section at the end
       - Format each source as: "Source Name: URL"
       - Prioritize credible, primary sources

    Create a polished, comprehensive response that maintains an appropriate level of detail and tone based on the confidence score and query type.
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
def run_search_agent(query: str, blacklist=None, debug_mode=False):
    """Run the search agent with a given query.

    Args:
        query (str): The search query to process
        blacklist (list, optional): List of domains to exclude from search results
        debug_mode (bool, optional): Enable detailed debugging output
    """
    # Create a session ID for tracking this search
    session_id = f"search_{int(time.time())}"

    print(f"\n{'=' * 80}")
    print(f"🔍 SEARCH SESSION: {session_id}")
    print(f"📝 QUERY: {query}")
    if blacklist:
        print(f"🚫 BLACKLISTED DOMAINS: {', '.join(blacklist)}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    initial_state = {
        "query": query,
        "query_type": None,
        "search_results": [],
        "search_queries": [],
        "blacklist": blacklist or [],
        "research_data": {},
        "verification_status": None,
        "verification_feedback": None,
        "confidence_score": None,
        "attempts": 0,
        "max_attempts": 3,  # Will be updated by query_assessment
        "final_answer": None,
        "sources": [],
        "messages": []
    }

    # Configure logging level based on debug mode
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        log_step("SYSTEM", "Debug mode enabled - showing detailed execution logs")
    else:
        logger.setLevel(logging.INFO)

    try:
        log_step("SYSTEM", f"Starting search process for: '{query}'")
        result = compiled_search_graph.invoke(initial_state)

        # Post-execution summary
        execution_time = time.time() - start_time
        confidence = result.get('confidence_score', 0.0)

        # Format confidence indicator with visual cue
        confidence_indicator = ""
        if confidence >= 0.8:
            confidence_indicator = "🟢 High confidence"
        elif confidence >= 0.4:
            confidence_indicator = "🟡 Medium confidence"
        else:
            confidence_indicator = "🔴 Low confidence"

        print(f"\n{'=' * 80}")
        print(f"📊 SEARCH SUMMARY:")
        print(f"{'=' * 80}")
        print(f"Session ID: {session_id}")
        print(f"Query: {query}")
        print(f"Strategy: {result['query_type']}")
        print(f"Iterations: {result['attempts']} of {result.get('max_attempts', 3)} maximum")
        print(f"Confidence: {confidence:.2f} ({confidence_indicator})")
        print(f"Execution time: {execution_time:.2f} seconds")

        # Count filtered sites
        filtered_count = 0
        for search_result in result.get("search_results", []):
            if "filtered_sites" in search_result:
                filtered_count += len(search_result["filtered_sites"])

        if filtered_count > 0:
            print(f"Blacklisted sites filtered: {filtered_count}")

        # Get top sources by relevance
        all_sites = set()
        top_sources = []

        if result.get("sources"):
            # Sort sources by position (relevance)
            sorted_sources = sorted(result.get("sources", []), key=lambda x: x.get("position", 999))

            # Get unique sites
            for source in sorted_sources:
                if "url" in source:
                    all_sites.add(source["url"])

                    # Collect top 5 sources with titles
                    if len(top_sources) < 5 and source.get("title"):
                        if source["url"] not in [s["url"] for s in top_sources]:
                            top_sources.append({
                                "title": source.get("title", "Unknown"),
                                "url": source["url"]
                            })

        if all_sites:
            print(f"\nSources: {len(all_sites)} unique sites consulted")
            if top_sources and debug_mode:
                print("Top sources:")
                for i, source in enumerate(top_sources, 1):
                    print(f"  {i}. {source['title']}")
                    print(f"     {source['url']}")

        print(f"\n{'=' * 80}")
        print(f"💡 ANSWER ({confidence_indicator}):")
        print(f"{'=' * 80}")

        return result["final_answer"]

    except Exception as e:
        log_step("ERROR", f"Search process failed: {str(e)}")
        print(f"\n{'=' * 80}")
        print(f"❌ ERROR: Search process failed")
        print(f"{'=' * 80}")
        print(f"An error occurred during the search process: {str(e)}")
        print(f"Please try again with a different query or check the logs for details.")
        return f"Search failed: {str(e)}"

def get_confidence_level(score):
    """Return a human-readable confidence level based on the score."""
    if score >= 0.8:
        return "High confidence"
    elif score >= 0.4:
        return "Medium confidence"
    else:
        return "Low confidence"

# Main functionality
def main():
    """CLI interface for the search agent"""
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph Search Agent")
    parser.add_argument("query", nargs="?", help="The search query")
    parser.add_argument("--blacklist", "-b", help="Comma-separated list of domains to blacklist", default="")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")

    args = parser.parse_args()

    if args.demo:
        run_demo(debug_mode=args.debug)
    elif args.query:
        blacklist = [domain.strip() for domain in args.blacklist.split(",")] if args.blacklist else None
        answer = run_search_agent(args.query, blacklist, debug_mode=args.debug)
        print(f"\n{answer}")
    else:
        parser.print_help()

def run_demo(debug_mode=False):
    """Run a demonstration of the search agent with sample queries"""
    print("\n" + "=" * 80)
    print("🚀 LANGGRAPH SEARCH AGENT DEMONSTRATION")
    print("=" * 80)
    print("\nThis demonstration will run the following types of queries:")
    print("1. Direct knowledge query (answered from LLM knowledge)")
    print("2. Simple search query (requires verification of current information)")
    print("3. Deep research query (requires multiple searches and synthesis)")
    print("4. Query with domain blacklisting (filters certain sites)\n")

    demo_queries = [
        ("What is the theory of relativity?", None, "DIRECT KNOWLEDGE QUERY"),
        ("Who won the Super Bowl in 2024?", None, "SIMPLE SEARCH QUERY"),
        ("What are the current global approaches to quantum computing research?", None, "DEEP RESEARCH QUERY"),
        ("What are popular coding tutorials for beginners?", ["youtube.com"], "BLACKLISTED DOMAIN QUERY")
    ]

    for query, blacklist, description in demo_queries:
        print("\n" + "=" * 80)
        print(f"📌 DEMO: {description}")
        print("=" * 80)
        start = time.time()
        answer = run_search_agent(query, blacklist, debug_mode=debug_mode)
        end = time.time()
        print(f"\nSearch completed in {end - start:.2f} seconds\n")
        print("-" * 80)

# Example usage
if __name__ == "__main__":
    main()