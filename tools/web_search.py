"""
Web Search Tool for Agentix agents.

This tool provides web search capabilities including:
- Search engine integration
- Web page content extraction
- URL validation and safety checks
- Result filtering and ranking
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
from pydantic import BaseModel, Field, validator

from .base import BaseTool, ToolConfig, ToolResult, ToolStatus


class WebSearchConfig(ToolConfig):
    """Configuration for web search tool."""
    
    # Search engine settings
    search_engine: str = "google"  # google, bing, duckduckgo
    api_key: Optional[str] = None
    search_engine_id: Optional[str] = None
    
    # Search parameters
    max_results: int = 10
    safe_search: bool = True
    language: str = "en"
    region: str = "us"
    
    # Content extraction
    extract_content: bool = True
    max_content_length: int = 5000
    
    # Filtering
    blocked_domains: List[str] = Field(default_factory=list)
    allowed_file_types: List[str] = Field(default_factory=lambda: ["html", "pdf", "txt"])
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if v < 1 or v > 100:
            raise ValueError("max_results must be between 1 and 100")
        return v


class SearchResult(BaseModel):
    """Individual search result."""
    
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    
    # Metadata
    domain: str
    file_type: str = "html"
    language: Optional[str] = None
    
    # Ranking
    rank: int
    relevance_score: float = 0.0
    
    # Timestamps
    published_date: Optional[datetime] = None
    crawled_date: datetime = Field(default_factory=datetime.now)


class WebSearchTool(BaseTool):
    """
    Web Search Tool for performing web searches and content extraction.
    
    This tool provides:
    - Multi-engine search support
    - Content extraction from web pages
    - Safety filtering and validation
    - Result ranking and relevance scoring
    """
    
    def __init__(self, config: WebSearchConfig):
        super().__init__(config)
        self.search_config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize search engine specific settings
        self._setup_search_engine()
    
    def _setup_search_engine(self):
        """Setup search engine specific configuration."""
        if self.search_config.search_engine == "google":
            if not self.search_config.api_key or not self.search_config.search_engine_id:
                self.logger.warning("Google search requires API key and search engine ID")
        
        elif self.search_config.search_engine == "bing":
            if not self.search_config.api_key:
                self.logger.warning("Bing search requires API key")
        
        # DuckDuckGo doesn't require API key
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute web search with the given parameters."""
        
        query = parameters.get('query', '')
        max_results = parameters.get('max_results', self.search_config.max_results)
        extract_content = parameters.get('extract_content', self.search_config.extract_content)
        
        if not query:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error="Search query is required",
                tool_name=self.name,
                tool_version=self.version
            )
        
        try:
            # Initialize HTTP session if needed
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Perform search
            search_results = await self._perform_search(query, max_results)
            
            # Extract content if requested
            if extract_content:
                search_results = await self._extract_content(search_results)
            
            # Filter and rank results
            filtered_results = self._filter_results(search_results)
            ranked_results = self._rank_results(filtered_results, query)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    'query': query,
                    'results': [result.dict() for result in ranked_results],
                    'total_results': len(ranked_results),
                    'search_engine': self.search_config.search_engine
                },
                metadata={
                    'search_timestamp': datetime.now().isoformat(),
                    'content_extracted': extract_content
                },
                tool_name=self.name,
                tool_version=self.version
            )
            
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Search failed: {str(e)}",
                tool_name=self.name,
                tool_version=self.version
            )
    
    async def _perform_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform the actual web search."""
        
        if self.search_config.search_engine == "google":
            return await self._google_search(query, max_results)
        elif self.search_config.search_engine == "bing":
            return await self._bing_search(query, max_results)
        elif self.search_config.search_engine == "duckduckgo":
            return await self._duckduckgo_search(query, max_results)
        else:
            raise ValueError(f"Unsupported search engine: {self.search_config.search_engine}")
    
    async def _google_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform Google Custom Search."""
        if not self.search_config.api_key or not self.search_config.search_engine_id:
            raise ValueError("Google search requires API key and search engine ID")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.search_config.api_key,
            'cx': self.search_config.search_engine_id,
            'q': query,
            'num': min(max_results, 10),  # Google API limit
            'safe': 'active' if self.search_config.safe_search else 'off',
            'lr': f'lang_{self.search_config.language}',
            'gl': self.search_config.region
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Google search API error: {response.status}")
            
            data = await response.json()
            results = []
            
            for i, item in enumerate(data.get('items', [])):
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    domain=self._extract_domain(item.get('link', '')),
                    rank=i + 1
                )
                results.append(result)
            
            return results
    
    async def _bing_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform Bing Web Search."""
        if not self.search_config.api_key:
            raise ValueError("Bing search requires API key")
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {'Ocp-Apim-Subscription-Key': self.search_config.api_key}
        params = {
            'q': query,
            'count': min(max_results, 50),  # Bing API limit
            'safeSearch': 'Strict' if self.search_config.safe_search else 'Off',
            'mkt': f'{self.search_config.language}-{self.search_config.region}'
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status != 200:
                raise Exception(f"Bing search API error: {response.status}")
            
            data = await response.json()
            results = []
            
            for i, item in enumerate(data.get('webPages', {}).get('value', [])):
                result = SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    domain=self._extract_domain(item.get('url', '')),
                    rank=i + 1
                )
                results.append(result)
            
            return results
    
    async def _duckduckgo_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform DuckDuckGo search (using instant answer API)."""
        # Note: This is a simplified implementation
        # For production use, consider using a proper DuckDuckGo API wrapper
        
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"DuckDuckGo search error: {response.status}")
            
            data = await response.json()
            results = []
            
            # DuckDuckGo instant answer API has limited web results
            # This is a placeholder implementation
            if data.get('AbstractText'):
                result = SearchResult(
                    title=data.get('Heading', query),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('AbstractText', ''),
                    domain=self._extract_domain(data.get('AbstractURL', '')),
                    rank=1
                )
                results.append(result)
            
            return results[:max_results]
    
    async def _extract_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Extract content from web pages."""
        
        for result in results:
            try:
                # Skip if domain is blocked
                if result.domain in self.search_config.blocked_domains:
                    continue
                
                # Extract content based on file type
                if result.file_type == "html":
                    content = await self._extract_html_content(result.url)
                    result.content = content[:self.search_config.max_content_length]
                
                # Add delay to be respectful to servers
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract content from {result.url}: {str(e)}")
                result.content = None
        
        return results
    
    async def _extract_html_content(self, url: str) -> str:
        """Extract text content from HTML page."""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return ""
                
                html = await response.text()
                
                # Simple text extraction (in production, use BeautifulSoup or similar)
                # This is a placeholder implementation
                import re
                
                # Remove script and style elements
                html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', '', html)
                
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                return text
                
        except Exception as e:
            self.logger.warning(f"HTML content extraction failed for {url}: {str(e)}")
            return ""
    
    def _filter_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter search results based on configuration."""
        filtered = []
        
        for result in results:
            # Skip blocked domains
            if result.domain in self.search_config.blocked_domains:
                continue
            
            # Check file type
            if result.file_type not in self.search_config.allowed_file_types:
                continue
            
            # Basic URL validation
            if not result.url.startswith(('http://', 'https://')):
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank and score search results."""
        query_terms = query.lower().split()
        
        for result in results:
            score = 0.0
            
            # Title relevance
            title_lower = result.title.lower()
            title_matches = sum(1 for term in query_terms if term in title_lower)
            score += (title_matches / len(query_terms)) * 0.4
            
            # Snippet relevance
            snippet_lower = result.snippet.lower()
            snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
            score += (snippet_matches / len(query_terms)) * 0.3
            
            # Content relevance (if available)
            if result.content:
                content_lower = result.content.lower()
                content_matches = sum(1 for term in query_terms if term in content_lower)
                score += (content_matches / len(query_terms)) * 0.2
            
            # Rank bonus (higher rank = lower bonus)
            rank_bonus = max(0, (11 - result.rank) / 10) * 0.1
            score += rank_bonus
            
            result.relevance_score = min(score, 1.0)
        
        # Sort by relevance score
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate search parameters."""
        query = parameters.get('query')
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return False
        
        max_results = parameters.get('max_results', self.search_config.max_results)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 100:
            return False
        
        return True
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None
