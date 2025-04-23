import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
from typing import List, Dict, Tuple, Set
import logging

class WebCrawler:
    """A web crawler that fetches links from pages and prioritizes them based on relevance."""

    def __init__(self, max_depth: int = 1, max_pages_per_domain: int = 5,
                 crawl_delay: float = 0.5, timeout: int = 10,
                 user_agent: str = "LangGraphSearchAgent/1.0"):
        """
        Initialize the web crawler.

        Args:
            max_depth: Maximum depth to crawl (default: 1)
            max_pages_per_domain: Maximum pages to crawl per domain (default: 5)
            crawl_delay: Delay between requests in seconds (default: 0.5)
            timeout: Request timeout in seconds (default: 10)
            user_agent: User agent to use for requests
        """
        self.max_depth = max_depth
        self.max_pages_per_domain = max_pages_per_domain
        self.crawl_delay = crawl_delay
        self.timeout = timeout
        self.user_agent = user_agent
        self.visited_urls = set()
        self.logger = logging.getLogger('search_agent.crawler')

    def get_domain(self, url: str) -> str:
        """Extract the domain from a URL."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Handle www prefix
        if domain.startswith('www.'):
            domain = domain[4:]

        return domain

    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs belong to the same domain."""
        return self.get_domain(url1) == self.get_domain(url2)

    def fetch_page(self, url: str) -> Tuple[bool, str]:
        """
        Fetch a page from a URL.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (success, content)
        """
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)

            # Check if the request was successful
            if response.status_code == 200:
                # Check if the content type is HTML
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    return True, response.text
                else:
                    self.logger.warning(f"Non-HTML content type: {content_type} for URL: {url}")
                    return False, f"Non-HTML content type: {content_type}"
            else:
                self.logger.warning(f"Failed to fetch URL: {url} - Status code: {response.status_code}")
                return False, f"HTTP error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error fetching URL: {url} - {str(e)}")
            return False, f"Request error: {str(e)}"

    def extract_links(self, base_url: str, html_content: str) -> List[str]:
        """
        Extract links from HTML content.

        Args:
            base_url: Base URL for resolving relative links
            html_content: HTML content to extract links from

        Returns:
            List of absolute URLs
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []

            # Extract links from <a> tags
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']

                # Skip anchors, javascript, and mailto links
                if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue

                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)

                # Skip URLs with fragments or query parameters to avoid duplicate content
                parsed_url = urlparse(absolute_url)
                cleaned_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

                links.append(cleaned_url)

            return links

        except Exception as e:
            self.logger.warning(f"Error extracting links from URL: {base_url} - {str(e)}")
            return []

    def rank_urls(self, urls: List[str], query: str) -> List[Tuple[str, float]]:
        """
        Rank URLs based on relevance to the query.

        Args:
            urls: List of URLs to rank
            query: Search query

        Returns:
            List of (URL, score) tuples sorted by score in descending order
        """
        # Extract query terms
        query_terms = re.findall(r'\w+', query.lower())
        scored_urls = []

        for url in urls:
            # Basic relevance scoring - check if query terms appear in URL
            parsed_url = urlparse(url)
            url_path = parsed_url.path.lower()

            # Count matches in the URL path
            score = 0.0
            for term in query_terms:
                if term in url_path:
                    # Higher weight for terms in path segments
                    score += 1.0

                    # Even higher weight for terms in the last path segment (likely the content name)
                    last_segment = url_path.split('/')[-1] if url_path and url_path != '/' else ''
                    if term in last_segment:
                        score += 0.5

            # Adjust score based on URL structure and features

            # Prefer shorter, cleaner URLs
            segments = [s for s in url_path.split('/') if s]
            if 1 <= len(segments) <= 3:
                score += 0.3

            # Prefer URLs with keywords like 'article', 'blog', 'research'
            content_indicators = ['article', 'blog', 'post', 'research', 'paper', 'study', 'guide', 'tutorial']
            for indicator in content_indicators:
                if indicator in url_path:
                    score += 0.2
                    break

            # Penalize URLs likely to be low-value
            low_value_indicators = ['comment', 'tag', 'category', 'search', 'archive', 'author', 'login', 'register']
            for indicator in low_value_indicators:
                if indicator in url_path:
                    score -= 0.3
                    break

            scored_urls.append((url, score))

        # Sort by score in descending order
        return sorted(scored_urls, key=lambda x: x[1], reverse=True)

    def extract_content(self, html: str) -> str:
        """
        Extract the main textual content from HTML.

        Args:
            html: HTML content

        Returns:
            Extracted main content text
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()

            # Extract text from paragraphs, headings, and list items
            content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            content = '\n'.join([elem.get_text().strip() for elem in content_elements])

            # Clean up the text (remove extra spaces and newlines)
            content = re.sub(r'\s+', ' ', content).strip()

            return content

        except Exception as e:
            self.logger.warning(f"Error extracting content from HTML: {str(e)}")
            return ""

    def extract_metadata(self, html: str, url: str) -> Dict:
        """
        Extract metadata from HTML.

        Args:
            html: HTML content
            url: Source URL

        Returns:
            Dictionary of metadata
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            metadata = {
                'url': url,
                'title': '',
                'description': '',
                'keywords': ''
            }

            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()

            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and 'content' in meta_desc.attrs:
                metadata['description'] = meta_desc['content'].strip()

            # Extract meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords and 'content' in meta_keywords.attrs:
                metadata['keywords'] = meta_keywords['content'].strip()

            return metadata

        except Exception as e:
            self.logger.warning(f"Error extracting metadata from URL: {url} - {str(e)}")
            return {'url': url, 'title': '', 'description': '', 'keywords': ''}

    def crawl_url(self, url: str, query: str) -> Dict:
        """
        Crawl a URL and extract content and related URLs.

        Args:
            url: URL to crawl
            query: Search query for ranking related URLs

        Returns:
            Dictionary with content, metadata, and related URLs
        """
        if url in self.visited_urls:
            return {'url': url, 'status': 'already_visited'}

        self.visited_urls.add(url)

        # Fetch the page
        success, content = self.fetch_page(url)
        if not success:
            return {'url': url, 'status': 'fetch_failed', 'error': content}

        # Extract content and metadata
        page_content = self.extract_content(content)
        metadata = self.extract_metadata(content, url)

        # Extract links from the same domain
        all_links = self.extract_links(url, content)
        same_domain_links = [link for link in all_links if self.is_same_domain(link, url)]

        # Rank the links
        ranked_links = self.rank_urls(same_domain_links, query)

        return {
            'url': url,
            'status': 'success',
            'content': page_content,
            'metadata': metadata,
            'links': same_domain_links,
            'ranked_links': ranked_links
        }

    def crawl_domain(self, start_url: str, query: str) -> Dict:
        """
        Crawl a domain starting from a URL.

        Args:
            start_url: Starting URL
            query: Search query for ranking URLs

        Returns:
            Dictionary with crawl results
        """
        self.logger.info(f"Starting domain crawl for: {start_url}")

        # Reset visit tracking for new crawl
        self.visited_urls = set()

        # Crawl the start URL
        start_result = self.crawl_url(start_url, query)

        if start_result.get('status') != 'success':
            self.logger.warning(f"Failed to crawl start URL: {start_url}")
            return {
                'root_url': start_url,
                'status': 'fail',
                'error': start_result.get('error', 'Unknown error'),
                'pages': []
            }

        # Initialize results structure
        results = {
            'root_url': start_url,
            'status': 'success',
            'domain': self.get_domain(start_url),
            'pages': [
                {
                    'url': start_url,
                    'content': start_result['content'],
                    'metadata': start_result['metadata']
                }
            ],
            'crawled_count': 1
        }

        # Get ranked links from start URL
        if not start_result.get('ranked_links'):
            self.logger.info(f"No links found at start URL: {start_url}")
            return results

        # Get top N URLs to crawl next
        urls_to_crawl = [url for url, score in start_result['ranked_links'][:self.max_pages_per_domain]]

        # Crawl selected URLs
        for url in urls_to_crawl:
            if len(results['pages']) >= self.max_pages_per_domain:
                break

            if url in self.visited_urls:
                continue

            # Add delay between requests
            time.sleep(self.crawl_delay)

            self.logger.info(f"Crawling linked URL: {url}")
            result = self.crawl_url(url, query)

            if result.get('status') == 'success':
                results['pages'].append({
                    'url': url,
                    'content': result['content'],
                    'metadata': result['metadata']
                })
                results['crawled_count'] += 1

        self.logger.info(f"Domain crawl complete. Crawled {results['crawled_count']} pages from {self.get_domain(start_url)}")
        return results


# Function to integrate with the search agent
def enhance_search_results(search_results, query, max_pages_per_domain=5):
    """
    Enhance search results by crawling the top results.

    Args:
        search_results: Original search results from the search API
        query: Original search query
        max_pages_per_domain: Maximum pages to crawl per domain

    Returns:
        Enhanced search results with crawled content
    """
    enhanced_results = {
        'original_results': search_results,
        'crawled_results': [],
        'total_crawled_pages': 0
    }

    logger = logging.getLogger('search_agent.crawler')
    crawler = WebCrawler(max_pages_per_domain=max_pages_per_domain)

    # Extract URLs from the search results
    urls_to_crawl = []
    if isinstance(search_results, dict) and "organic" in search_results:
        for result in search_results["organic"]:
            if "link" in result:
                urls_to_crawl.append(result["link"])

    if not urls_to_crawl:
        logger.warning("No URLs found in search results to crawl")
        return enhanced_results

    # Limit to top 5 URLs
    top_urls = urls_to_crawl[:5]
    logger.info(f"Starting crawl for top {len(top_urls)} search result URLs")

    # Crawl each domain
    for url in top_urls:
        logger.info(f"Crawling domain from URL: {url}")
        domain_results = crawler.crawl_domain(url, query)

        if domain_results.get('status') == 'success':
            enhanced_results['crawled_results'].append(domain_results)
            enhanced_results['total_crawled_pages'] += domain_results.get('crawled_count', 0)

    logger.info(f"Search enhancement complete. Crawled {enhanced_results['total_crawled_pages']} pages from {len(enhanced_results['crawled_results'])} domains")
    return enhanced_results