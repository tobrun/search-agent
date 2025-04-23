from typing import TypedDict, List, Dict, Optional, Annotated, Any, Literal, Type, Union
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities import (
    GoogleSerperAPIWrapper,
    BingSearchAPIWrapper,
    BraveSearchWrapper,
    DuckDuckGoSearchAPIWrapper,
    MojeekSearchAPIWrapper,
    SearxSearchWrapper,
    YouSearchAPIWrapper
)
from langchain_community.utilities.jina_search import JinaSearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import time
import os
import logging
import json
import re
from datetime import datetime
from urllib.parse import urlparse
from web_crawler import WebCrawler, enhance_search_results

# Available search engines
SEARCH_ENGINES = {
    "google_serper": GoogleSerperAPIWrapper,
    "bing": BingSearchAPIWrapper,
    "brave": BraveSearchWrapper,
    "duckduckgo": DuckDuckGoSearchAPIWrapper,
    "jina": JinaSearchAPIWrapper,
    "mojeek": MojeekSearchAPIWrapper,
    "searx": SearxSearchWrapper,
    "tavily": TavilySearchAPIWrapper,
    "you": YouSearchAPIWrapper
}

# Environment variable prefix for search engine configuration
ENV_PREFIX = "SEARCH_"

# Directory for storing temp data
TEMP_DIR = ".tmp"

# Required API keys for each search engine
REQUIRED_API_KEYS = {
    "google_serper": ["SERPER_API_KEY"],
    "bing": ["BING_SUBSCRIPTION_KEY"],
    "brave": ["BRAVE_API_KEY"],
    "tavily": ["TAVILY_API_KEY"],
    "you": ["YOU_API_KEY"]
    # Other engines may not require API keys or use different env vars
}

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
llm = ChatOpenAI(
        temperature=0.2,
        openai_api_base="http://192.168.0.172:8000/v1",
        model="gemma-3-27b-it"
    )

# Configure and initialize search engine
def init_search_engine(engine_name: str = "google_serper", **kwargs) -> Any:
    """Initialize a search engine by name with optional configuration parameters.

    Args:
        engine_name: Name of the search engine to use (must be in SEARCH_ENGINES)
        **kwargs: Configuration parameters to pass to the search engine constructor

    Returns:
        Initialized search engine instance

    Raises:
        ValueError: If engine_name is not recognized or required API keys are missing
    """
    if engine_name not in SEARCH_ENGINES:
        available_engines = ", ".join(SEARCH_ENGINES.keys())
        raise ValueError(f"Unknown search engine '{engine_name}'. Available engines: {available_engines}")

    # Check for required API keys in environment variables
    if engine_name in REQUIRED_API_KEYS:
        for key_name in REQUIRED_API_KEYS[engine_name]:
            if key_name not in os.environ and key_name not in kwargs:
                raise ValueError(f"Missing required API key {key_name} for {engine_name}")

    # Look for engine-specific configuration in environment variables
    engine_env_prefix = f"{ENV_PREFIX}{engine_name.upper()}_"
    for env_key, env_value in os.environ.items():
        if env_key.startswith(engine_env_prefix):
            # Convert environment variable to parameter name by removing prefix and converting to lowercase
            param_name = env_key[len(engine_env_prefix):].lower()
            if param_name not in kwargs:  # Don't override explicit parameters
                kwargs[param_name] = env_value

    # Handle common API key environment variables
    if engine_name == "google_serper" and "api_key" not in kwargs and "SERPER_API_KEY" in os.environ:
        kwargs["api_key"] = os.environ["SERPER_API_KEY"]
    elif engine_name == "bing" and "subscription_key" not in kwargs and "BING_SUBSCRIPTION_KEY" in os.environ:
        kwargs["subscription_key"] = os.environ["BING_SUBSCRIPTION_KEY"]
    elif engine_name == "brave" and "api_key" not in kwargs and "BRAVE_API_KEY" in os.environ:
        kwargs["api_key"] = os.environ["BRAVE_API_KEY"]
    elif engine_name == "tavily" and "api_key" not in kwargs and "TAVILY_API_KEY" in os.environ:
        kwargs["api_key"] = os.environ["TAVILY_API_KEY"]
    elif engine_name == "you" and "api_key" not in kwargs and "YOU_API_KEY" in os.environ:
        kwargs["api_key"] = os.environ["YOU_API_KEY"]
    # Handle DuckDuckGo default parameters if not provided
    elif engine_name == "duckduckgo" and "max_results" not in kwargs:
        kwargs["max_results"] = int(os.environ.get(f"{engine_env_prefix}MAX_RESULTS", "10"))

    # Initialize the engine
    engine_class = SEARCH_ENGINES[engine_name]
    log_step("SYSTEM", f"Initializing {engine_name} search engine")
    return engine_class(**kwargs)

# Default search tool (can be overridden when calling run_search_agent)
search_tool = None  # Will be initialized on first use

def get_search_tool(engine_name="google_serper", **kwargs):
    """Get or initialize the search tool"""
    global search_tool
    if search_tool is None:
        search_tool = init_search_engine(engine_name, **kwargs)
    return search_tool

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

# Define a wrapper function to handle different search engines with varying interfaces
def search_with_engine(search_tool, query, **kwargs):
    """Execute search with the configured search engine, handling differences in interfaces.

    Args:
        search_tool: The initialized search engine
        query: The search query
        **kwargs: Additional parameters for the search

    Returns:
        Search results in a standardized format
    """
    # Get the engine name
    engine_name = getattr(search_tool, "__class__", None)
    engine_name = engine_name.__name__ if engine_name else "Unknown"

    try:
        # Handle engine-specific search methods
        if "DuckDuckGoSearchAPIWrapper" in engine_name:
            # DuckDuckGo needs max_results
            max_results = kwargs.get("max_results", 10)
            results = search_tool.results(query, max_results=max_results)
        elif "TavilySearchAPIWrapper" in engine_name:
            # Tavily has a specific format and may need additional parameters
            results = search_tool.results(query,
                                         max_results=kwargs.get("max_results", 10),
                                         search_depth=kwargs.get("search_depth", "basic"))
            # Standardize the output format to match other engines
            if isinstance(results, list):
                return {"organic": results}
            return results
        else:
            # Default method for most engines
            results = search_tool.results(query)

        # Standardize results format
        if not isinstance(results, dict):
            results = {"results": results}

        # If results has no "organic" key but has other data, add empty organic
        if "organic" not in results and len(results) > 0:
            results["organic"] = []

        return results
    except Exception as e:
        log_step("SEARCH_ERROR", f"Error with {engine_name}: {str(e)}")
        # Return standardized empty results
        return {"organic": [], "error": str(e)}

# Function to save data to JSON file
def save_to_json(data, session_id, filename):
    """Save data to a JSON file in the session's directory.

    Args:
        data: Data to save (dict or list)
        session_id: Session ID for directory naming
        filename: Name of the file to save

    Returns:
        Path to the saved file
    """
    # Create the session directory if it doesn't exist
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(session_dir, filename)

    # Save the data to JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log_step("STORAGE", f"Saved data to {file_path}")
    return file_path

# Function to load data from JSON file
def load_from_json(session_id, filename):
    """Load data from a JSON file in the session's directory.

    Args:
        session_id: Session ID for directory naming
        filename: Name of the file to load

    Returns:
        Loaded data or None if file doesn't exist
    """
    file_path = os.path.join(TEMP_DIR, session_id, filename)

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    log_step("STORAGE", f"Loaded data from {file_path}")
    return data

def execute_search(state: SearchState) -> Dict:
    """Execute a search operation using the generated search query and enhance with web crawling."""
    search_queries = state["search_queries"]
    latest_query = search_queries[-1]
    blacklist = state.get("blacklist", [])
    enable_crawling = state.get("enable_crawling", True)
    search_engine = state.get("search_engine", "google_serper")
    session_id = state.get("session_id", f"search_{int(time.time())}")

    log_step("EXECUTE_SEARCH", f"Searching for: '{latest_query}'")
    if blacklist:
        log_step("EXECUTE_SEARCH", f"Using domain blacklist: {blacklist}")

    try:
        # Get the configured search tool
        search_tool = get_search_tool()

        # First, try with quotes for exact matching
        quoted_query = latest_query
        if not (quoted_query.startswith('"') and quoted_query.endswith('"')):
            quoted_query = f'"{latest_query}"'

        log_step("EXECUTE_SEARCH", f"Trying exact search with: {quoted_query}")
        search_results = search_with_engine(search_tool, quoted_query, max_results=10)

        # Save the exact search results
        save_to_json(
            search_results,
            session_id,
            f"search_exact_{len(search_queries)}.json"
        )

        organic_results_count = len(search_results.get("organic", []))
        log_step("EXECUTE_SEARCH", f"Exact search returned {organic_results_count} results")

        # If no results with quotes, retry without quotes
        if organic_results_count == 0:
            unquoted_query = latest_query.strip('"')
            log_step("EXECUTE_SEARCH", f"No results with exact search, trying broader search: {unquoted_query}")
            search_results = search_with_engine(search_tool, unquoted_query, max_results=10)

            # Save the broader search results
            save_to_json(
                search_results,
                session_id,
                f"search_broad_{len(search_queries)}.json"
            )

            organic_results_count = len(search_results.get("organic", []))
            log_step("EXECUTE_SEARCH", f"Broader search returned {organic_results_count} results")

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

        # Initialize crawling data
        crawled_data = {}

        # Enhance results with web crawling if enabled and we have results
        if enable_crawling and len(sites) > 0:
            log_step("EXECUTE_SEARCH", "Enhancing search results with web crawling...")

            original_query = state["query"]  # Use the original user query for relevance
            enhanced_results = enhance_search_results(search_results, original_query)

            # Save the enhanced results with crawled data
            save_to_json(
                enhanced_results,
                session_id,
                f"search_enhanced_{len(search_queries)}.json"
            )

            # Save raw crawled content for each domain
            for domain_data in enhanced_results.get("crawled_results", []):
                domain = domain_data.get("domain", "unknown")
                domain_pages = domain_data.get("pages", [])

                if domain_pages:
                    save_to_json(
                        domain_pages,
                        session_id,
                        f"crawled_{domain.replace('.', '_')}_{len(search_queries)}.json"
                    )

            # Log crawling results
            crawled_domains = len(enhanced_results.get('crawled_results', []))
            total_pages = enhanced_results.get('total_crawled_pages', 0)
            log_step("EXECUTE_SEARCH", f"Crawling enhanced results with {total_pages} pages from {crawled_domains} domains")

            # Update search results with enhanced data
            search_results["enhanced"] = enhanced_results
            crawled_data = enhanced_results

            # Add crawled pages to sources
            for domain_result in enhanced_results.get('crawled_results', []):
                domain = domain_result.get('domain', '')
                for page in domain_result.get('pages', []):
                    url = page.get('url', '')
                    metadata = page.get('metadata', {})

                    if url and url not in [s["url"] for s in sources]:
                        site_info = {
                            "url": url,
                            "title": metadata.get('title', f"Crawled page from {domain}"),
                            "snippet": metadata.get('description', ''),
                            "query": original_query,
                            "position": 100,  # Lower position for crawled results
                            "crawled": True
                        }
                        sources.append(site_info)

            log_step("EXECUTE_SEARCH", f"Sources list updated with crawled pages. Now contains {len(sources)} sources")

        # Save the final sources list
        save_to_json(
            sources,
            session_id,
            f"sources_{len(search_queries)}.json"
        )

        # Add to existing results
        all_results = state.get("search_results", [])
        all_results.append({
            "query": latest_query,
            "results": search_results,
            "crawled_data": crawled_data
        })

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
        error_info = {"query": latest_query, "results": [], "error": str(e)}

        # Save error information
        save_to_json(
            error_info,
            session_id,
            f"search_error_{len(search_queries)}.json"
        )

        return {
            "search_results": state.get("search_results", []) + [error_info],
            "messages": [
                {"role": "system", "content": f"Search error: {str(e)}"}
            ]
        }

def execute_parallel_searches(state: SearchState) -> Dict:
    """Execute multiple searches in parallel for deep research."""
    query = state["query"]
    blacklist = state.get("blacklist", [])
    search_engine = state.get("search_engine", "google_serper")
    session_id = state.get("session_id", f"search_{int(time.time())}")

    log_step("DEEP_RESEARCH", f"Generating multiple search queries for deep research: '{query}'")
    if blacklist:
        log_step("DEEP_RESEARCH", f"Using domain blacklist: {blacklist}")

    # Generate multiple search queries for different aspects - we want 4 additional queries
    # (plus the original query makes 5 total)
    query_generation_prompt = f"""
    For the deep research query: "{query}"

    Generate 4 different search queries that would help gather comprehensive information.
    Make each query focus on a different aspect or perspective of the topic.
    The original query will also be used, so make these complementary and diverse.
    Return only the 4 queries, one per line, no explanations.
    """

    messages = [HumanMessage(content=query_generation_prompt)]
    response = llm.invoke(messages)

    # Extract the search queries
    additional_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]

    # Ensure we have exactly 4 additional queries (pad if needed)
    while len(additional_queries) < 4:
        if len(additional_queries) < 1:
            additional_queries.append(query)  # Just repeat the original query
        else:
            # Generate variations by adding qualifiers
            additional_queries.append(f"{additional_queries[-1]} latest research")

    # If somehow we got more than 4, trim the list
    additional_queries = additional_queries[:4]

    # Add the original query at the beginning
    search_queries = [query] + additional_queries

    # Save the generated queries
    save_to_json(
        {"original_query": query, "search_queries": search_queries},
        session_id,
        "deep_research_queries.json"
    )

    log_step("DEEP_RESEARCH", f"Using 5 search queries for deep research:", search_queries)

    # If we have existing queries, add them
    existing_queries = state.get("search_queries", [])
    all_queries = existing_queries + search_queries

    # Get the configured search tool
    search_tool = get_search_tool()

    # Execute searches sequentially (instead of in parallel)
    search_results = []
    all_sites = []
    filtered_sites = []
    sources = state.get("sources", [])

    for i, query in enumerate(search_queries):
        log_step("DEEP_RESEARCH", f"Executing search for: '{query}'")
        try:
            results = search_with_engine(search_tool, query, max_results=10)

            # Save individual search results
            save_to_json(
                results,
                session_id,
                f"deep_search_{i+1}.json"
            )

            # Extract searched sites for logging and source tracking
            sites = []
            query_filtered_sites = []

            if isinstance(results, dict) and "organic" in results:
                # Filter out blacklisted domains
                filtered_results = []
                for j, result in enumerate(results["organic"]):
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
                            "position": j + 1
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

            # Enhance results with web crawling if enabled and we have results
            if state.get("enable_crawling", True) and len(sites) > 0:
                log_step("DEEP_RESEARCH", f"Enhancing results for '{query}' with web crawling...")

                enhanced_results = enhance_search_results(results, query)

                # Save the enhanced results with crawled data
                save_to_json(
                    enhanced_results,
                    session_id,
                    f"deep_enhanced_{i+1}.json"
                )

                # Save raw crawled content for each domain
                for domain_data in enhanced_results.get("crawled_results", []):
                    domain = domain_data.get("domain", "unknown")
                    domain_pages = domain_data.get("pages", [])

                    if domain_pages:
                        save_to_json(
                            domain_pages,
                            session_id,
                            f"deep_crawled_{domain.replace('.', '_')}_{i+1}.json"
                        )

                # Update results with enhanced data
                results["enhanced"] = enhanced_results

                # Add crawled pages to sources
                for domain_result in enhanced_results.get('crawled_results', []):
                    domain = domain_result.get('domain', '')
                    for page in domain_result.get('pages', []):
                        url = page.get('url', '')
                        metadata = page.get('metadata', {})

                        if url and url not in [s["url"] for s in sources]:
                            site_info = {
                                "url": url,
                                "title": metadata.get('title', f"Crawled page from {domain}"),
                                "snippet": metadata.get('description', ''),
                                "query": query,
                                "position": 100,  # Lower position for crawled results
                                "crawled": True
                            }
                            sources.append(site_info)

            search_results.append({"query": query, "results": results})
        except Exception as e:
            log_step("DEEP_RESEARCH", f"Search error for '{query}': {str(e)}")

            # Save error information
            error_info = {"query": query, "results": [], "error": str(e)}
            save_to_json(
                error_info,
                session_id,
                f"deep_search_error_{i+1}.json"
            )

            search_results.append(error_info)

    # Save final sources and filtered sites
    save_to_json(
        sources,
        session_id,
        "deep_research_sources.json"
    )

    save_to_json(
        {"filtered_sites": filtered_sites, "all_sites": all_sites},
        session_id,
        "deep_research_sites.json"
    )

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
    crawled_content_by_domain = {}

    for result_set in search_results:
        if result_set.get("results"):
            all_results.append({
                "query": result_set["query"],
                "results": result_set["results"]
            })

            # Extract crawled content if available
            if "enhanced" in result_set.get("results", {}):
                enhanced = result_set["results"]["enhanced"]
                for domain_data in enhanced.get("crawled_results", []):
                    domain = domain_data.get("domain", "unknown")

                    if domain not in crawled_content_by_domain:
                        crawled_content_by_domain[domain] = []

                    for page in domain_data.get("pages", []):
                        if page.get("content"):
                            crawled_content_by_domain[domain].append({
                                "url": page.get("url", ""),
                                "title": page.get("metadata", {}).get("title", "Untitled"),
                                "content": page.get("content", "")
                            })

    if not all_results:
        return {
            "messages": [
                {"role": "system", "content": "No valid results to aggregate."}
            ]
        }

    # Prepare crawled content summary
    crawled_content_summary = ""
    if crawled_content_by_domain:
        crawled_content_summary = "\nCrawled Content Summary:\n"
        for domain, pages in crawled_content_by_domain.items():
            crawled_content_summary += f"\nContent from {domain} ({len(pages)} pages):\n"

            for i, page in enumerate(pages):
                title = page.get("title", "Untitled")
                url = page.get("url", "")

                # Get a concise summary from the content
                content = page.get("content", "")
                content_summary = content[:300] + "..." if len(content) > 300 else content

                crawled_content_summary += f"Page {i+1}: {title} - {url}\n"
                crawled_content_summary += f"Content summary: {content_summary}\n\n"

    prompt = f"""
    Analyze these search results for the in-depth research query: "{query}"

    Search Results:
    {all_results}

    {crawled_content_summary if crawled_content_by_domain else ""}

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
def run_search_agent(query: str, blacklist=None, debug_mode=False, enable_crawling=True,
                     search_engine="google_serper", search_engine_kwargs=None):
    """Run the search agent with a given query.

    Args:
        query (str): The search query to process
        blacklist (list, optional): List of domains to exclude from search results
        debug_mode (bool, optional): Enable detailed debugging output
        enable_crawling (bool, optional): Enable web crawling to enhance search results
        search_engine (str, optional): Name of search engine to use (default: "google_serper")
        search_engine_kwargs (dict, optional): Additional configuration for the search engine
    """
    # Create a session ID for tracking this search
    session_id = f"search_{int(time.time())}"

    print(f"\n{'=' * 80}")
    print(f"🔍 SEARCH SESSION: {session_id}")
    print(f"📝 QUERY: {query}")
    if blacklist:
        print(f"🚫 BLACKLISTED DOMAINS: {', '.join(blacklist)}")
    if enable_crawling:
        print(f"🌐 WEB CRAWLING: Enabled (1 level deep, max 5 pages per domain)")
    print(f"🔎 SEARCH ENGINE: {search_engine}")
    print(f"💾 DATA DIRECTORY: {os.path.join(TEMP_DIR, session_id)}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    # Initialize the configured search engine
    global search_tool
    search_engine_kwargs = search_engine_kwargs or {}
    search_tool = init_search_engine(search_engine, **search_engine_kwargs)

    # Create the session directory
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

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
        "enable_crawling": enable_crawling,  # Add crawling flag to state
        "search_engine": search_engine,      # Add search engine to state
        "session_id": session_id,            # Add session ID to state
        "messages": []
    }

    # Save initial state
    save_to_json(
        {k: v for k, v in initial_state.items() if k != "messages"},
        session_id,
        "initial_state.json"
    )

    # Configure logging level based on debug mode
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        log_step("SYSTEM", "Debug mode enabled - showing detailed execution logs")
    else:
        logger.setLevel(logging.INFO)

    try:
        log_step("SYSTEM", f"Starting search process for: '{query}'")
        result = compiled_search_graph.invoke(initial_state)

        # Save final result
        save_to_json(
            {k: v for k, v in result.items() if k != "messages"},
            session_id,
            "final_result.json"
        )

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
        print(f"Data directory: {session_dir}")

        # Count filtered sites
        filtered_count = 0
        for search_result in result.get("search_results", []):
            if "filtered_sites" in search_result:
                filtered_count += len(search_result["filtered_sites"])

        if filtered_count > 0:
            print(f"Blacklisted sites filtered: {filtered_count}")

        # Count crawled pages
        total_crawled = 0
        crawled_domains = 0

        if enable_crawling:
            for search_result in result.get("search_results", []):
                if "results" in search_result and "enhanced" in search_result["results"]:
                    enhanced = search_result["results"]["enhanced"]
                    total_crawled += enhanced.get("total_crawled_pages", 0)
                    crawled_domains += len(enhanced.get("crawled_results", []))

            if total_crawled > 0:
                print(f"Web crawling: {total_crawled} pages from {crawled_domains} domains")

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
                                "url": source["url"],
                                "crawled": source.get("crawled", False)
                            })

        if all_sites:
            print(f"\nSources: {len(all_sites)} unique sites consulted")
            if top_sources:
                print("Top sources:")
                for i, source in enumerate(top_sources, 1):
                    crawled_indicator = " (crawled)" if source.get("crawled") else ""
                    print(f"  {i}. {source['title']}{crawled_indicator}")
                    print(f"     {source['url']}")

        print(f"\n{'=' * 80}")
        print(f"💡 ANSWER ({confidence_indicator}):")
        print(f"{'=' * 80}")
        print(f"\n{result['final_answer']}")

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
    parser.add_argument("--no-crawl", action="store_true", help="Disable web crawling")
    parser.add_argument("--search-engine", "-s", help="Search engine to use",
                       choices=list(SEARCH_ENGINES.keys()), default="google_serper")
    parser.add_argument("--api-key", "-k", help="API key for the search engine (overrides environment variable)")

    args = parser.parse_args()
    enable_crawling = not args.no_crawl

    # Set up search engine kwargs
    search_engine_kwargs = {}
    if args.api_key:
        search_engine_kwargs["api_key"] = args.api_key

    if args.demo:
        run_demo(debug_mode=args.debug, enable_crawling=enable_crawling,
                search_engine=args.search_engine, search_engine_kwargs=search_engine_kwargs)
    elif args.query:
        blacklist = [domain.strip() for domain in args.blacklist.split(",")] if args.blacklist else None
        answer = run_search_agent(args.query, blacklist, debug_mode=args.debug,
                                 enable_crawling=enable_crawling, search_engine=args.search_engine,
                                 search_engine_kwargs=search_engine_kwargs)
        print(f"\n{answer}")
    else:
        parser.print_help()

def run_demo(debug_mode=False, enable_crawling=True, search_engine="google_serper",
            search_engine_kwargs=None):
    """Run a demonstration of the search agent with sample queries"""
    print("\n" + "=" * 80)
    print("🚀 LANGGRAPH SEARCH AGENT DEMONSTRATION")
    print("=" * 80)
    print("\nThis demonstration will run the following types of queries:")
    print("1. Direct knowledge query (answered from LLM knowledge)")
    print("2. Simple search query (requires verification of current information)")
    print("3. Deep research query (requires multiple searches and synthesis)")
    print("4. Query with domain blacklisting (filters certain sites)")
    print(f"\n🔎 Using search engine: {search_engine}")
    if enable_crawling:
        print("\n🌐 Web crawling is ENABLED - results will be enhanced with content from related pages")
    else:
        print("\n🌐 Web crawling is DISABLED")
    print()

    demo_queries = [
        ("What will the launch price be in Belgium for the nintendo switch 2?", None, "SIMPLE SEARCH QUERY"),

        # ("What are the opening hours of Pisco y Nazca Ceviche Gastrobar in Washington DC?", None, "SIMPLE SEARCH QUERY"),
        # ("What are the opening hours of Pisco y Nazca Ceviche Gastrobar in Washington DC?", None, "DEEP RESEARCH QUERY")
        # ("What is the theory of relativity?", None, "DIRECT KNOWLEDGE QUERY"),
        # ("Who won the Super Bowl in 2024?", None, "SIMPLE SEARCH QUERY"),
        # ("What are the current global approaches to quantum computing research?", None, "DEEP RESEARCH QUERY"),
        # ("What are popular coding tutorials for beginners?", ["youtube.com"], "BLACKLISTED DOMAIN QUERY")
    ]

    for query, blacklist, description in demo_queries:
        print("\n" + "=" * 80)
        print(f"📌 DEMO: {description}")
        print("=" * 80)
        start = time.time()
        answer = run_search_agent(query, blacklist, debug_mode=debug_mode,
                                 enable_crawling=enable_crawling, search_engine=search_engine,
                                 search_engine_kwargs=search_engine_kwargs)
        end = time.time()
        print(f"\nSearch completed in {end - start:.2f} seconds\n")
        print("-" * 80)

# Example usage
if __name__ == "__main__":
    main()