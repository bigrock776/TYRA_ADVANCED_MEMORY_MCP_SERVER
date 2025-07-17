"""
Natural Language Parser for Web Crawling Commands.

Parses natural language requests into structured crawl parameters
for the Crawl4AI integration.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ...ingest.crawl4ai_runner import CrawlStrategy, ContentType
from ..utils.simple_logger import get_logger

logger = get_logger(__name__)


class ParsedIntent(str, Enum):
    """Types of crawling intents."""
    CRAWL_SINGLE = "crawl_single"
    CRAWL_SITE = "crawl_site"
    CRAWL_DOCUMENTATION = "crawl_documentation"
    CRAWL_RESEARCH = "crawl_research"
    CRAWL_NEWS = "crawl_news"
    CRAWL_BLOG = "crawl_blog"
    CRAWL_RECURSIVE = "crawl_recursive"
    UNKNOWN = "unknown"


@dataclass
class CrawlCommand:
    """Parsed crawl command structure."""
    urls: List[str]
    intent: ParsedIntent
    strategy: CrawlStrategy
    content_type: ContentType
    max_pages: Optional[int] = None
    depth: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None
    store_in_memory: bool = True
    extract_entities: bool = True
    confidence: float = 0.8


class NaturalLanguageCrawlParser:
    """
    Parser for converting natural language crawl requests into structured commands.
    
    Examples:
    - "Crawl the Wikipedia page about machine learning"
    - "Read the entire documentation at docs.python.org"
    - "Get me all blog posts from medium.com about AI"
    - "Research arxiv papers on transformer architectures"
    """
    
    def __init__(self):
        # Intent patterns
        self.intent_patterns = {
            ParsedIntent.CRAWL_SINGLE: [
                r"crawl (?:the )?(?:page|article|post)",
                r"read (?:the )?(?:page|article|post)",
                r"get (?:the )?(?:page|article|post)",
                r"fetch (?:the )?(?:page|article|post)",
                r"save (?:the )?(?:page|article|post)",
            ],
            ParsedIntent.CRAWL_SITE: [
                r"crawl (?:the )?(?:entire )?(?:site|website)",
                r"read (?:the )?(?:entire )?(?:site|website)",
                r"index (?:the )?(?:entire )?(?:site|website)",
                r"scrape (?:the )?(?:entire )?(?:site|website)",
            ],
            ParsedIntent.CRAWL_DOCUMENTATION: [
                r"(?:crawl|read|get) (?:the )?(?:documentation|docs)",
                r"fetch (?:all )?(?:the )?(?:documentation|docs)",
                r"save (?:the )?(?:documentation|docs)",
            ],
            ParsedIntent.CRAWL_RESEARCH: [
                r"research (?:papers?|articles?)",
                r"find (?:research )?papers?",
                r"get (?:academic|scientific) (?:papers?|articles?)",
                r"crawl arxiv",
                r"search (?:for )?papers?",
            ],
            ParsedIntent.CRAWL_NEWS: [
                r"(?:crawl|get|fetch) (?:the )?(?:latest )?news",
                r"read (?:recent|latest) (?:news|articles?)",
                r"find news (?:about|on)",
            ],
            ParsedIntent.CRAWL_BLOG: [
                r"(?:crawl|get|fetch) (?:blog )?posts?",
                r"read (?:the )?blog",
                r"find (?:blog )?(?:posts?|articles?)",
            ],
            ParsedIntent.CRAWL_RECURSIVE: [
                r"crawl (?:recursively|deeply)",
                r"follow (?:all )?links",
                r"spider (?:the )?(?:site|website)",
            ],
        }
        
        # URL extraction patterns
        self.url_patterns = [
            # Standard URLs
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            # URLs with "at" or "from"
            r'(?:at|from)\s+([\w.-]+\.(?:com|org|net|edu|gov|io|dev|ai|co|uk)(?:/[^\s]*)?)',
            # Domain mentions
            r'(?:on|about)\s+([\w.-]+\.(?:com|org|net|edu|gov|io|dev|ai|co|uk))',
        ]
        
        # Depth/limit patterns
        self.depth_patterns = [
            r'(?:up to |max(?:imum)? |limit to )?(\d+)\s*(?:levels?|deep|depth)',
            r'(?:depth|level)\s*(?:of\s*)?(\d+)',
        ]
        
        self.page_limit_patterns = [
            r'(?:up to |max(?:imum)? |limit to )?(\d+)\s*(?:pages?|articles?|posts?)',
            r'(?:first |top )?(\d+)\s*(?:pages?|articles?|posts?)',
        ]
        
        # Content type indicators
        self.content_indicators = {
            ContentType.DOCUMENTATION: ['documentation', 'docs', 'api reference', 'guide'],
            ContentType.RESEARCH_PAPER: ['paper', 'research', 'arxiv', 'academic', 'scientific'],
            ContentType.NEWS: ['news', 'latest', 'recent', 'breaking'],
            ContentType.BLOG_POST: ['blog', 'post', 'article', 'tutorial'],
            ContentType.FORUM_POST: ['forum', 'discussion', 'thread', 'stackoverflow'],
            ContentType.PRODUCT_PAGE: ['product', 'shop', 'store', 'buy'],
        }
    
    def parse(self, text: str) -> CrawlCommand:
        """
        Parse natural language text into a crawl command.
        
        Args:
            text: Natural language crawl request
            
        Returns:
            Parsed CrawlCommand object
        """
        text_lower = text.lower()
        
        # Extract URLs
        urls = self._extract_urls(text)
        
        # Determine intent
        intent = self._determine_intent(text_lower)
        
        # Determine strategy based on intent
        strategy = self._intent_to_strategy(intent)
        
        # Determine content type
        content_type = self._determine_content_type(text_lower)
        
        # Extract depth/limits
        max_pages = self._extract_page_limit(text_lower)
        depth = self._extract_depth(text_lower)
        
        # Extract filters
        filters = self._extract_filters(text_lower)
        
        # Determine if we should store in memory
        store_in_memory = "don't save" not in text_lower and "no memory" not in text_lower
        
        # Determine if we should extract entities
        extract_entities = "no entities" not in text_lower
        
        # Calculate confidence based on how well we parsed
        confidence = self._calculate_confidence(
            urls=urls,
            intent=intent,
            has_limits=bool(max_pages or depth)
        )
        
        return CrawlCommand(
            urls=urls,
            intent=intent,
            strategy=strategy,
            content_type=content_type,
            max_pages=max_pages,
            depth=depth,
            filters=filters,
            store_in_memory=store_in_memory,
            extract_entities=extract_entities,
            confidence=confidence
        )
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        urls = []
        
        for pattern in self.url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Handle different match types
                if isinstance(match, tuple):
                    url = match[0]
                else:
                    url = match
                
                # Add protocol if missing
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                # Clean up URL
                url = url.rstrip('.,;:')
                
                if url not in urls:
                    urls.append(url)
        
        return urls
    
    def _determine_intent(self, text: str) -> ParsedIntent:
        """Determine the crawling intent from text."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        
        # Default based on keywords
        if any(word in text for word in ['entire', 'all', 'whole', 'complete']):
            return ParsedIntent.CRAWL_SITE
        elif any(word in text for word in ['recursively', 'deep', 'follow']):
            return ParsedIntent.CRAWL_RECURSIVE
        
        return ParsedIntent.CRAWL_SINGLE
    
    def _intent_to_strategy(self, intent: ParsedIntent) -> CrawlStrategy:
        """Convert intent to crawl strategy."""
        mapping = {
            ParsedIntent.CRAWL_SINGLE: CrawlStrategy.SINGLE_PAGE,
            ParsedIntent.CRAWL_SITE: CrawlStrategy.SITE_MAP,
            ParsedIntent.CRAWL_DOCUMENTATION: CrawlStrategy.SITE_MAP,
            ParsedIntent.CRAWL_RECURSIVE: CrawlStrategy.RECURSIVE,
            ParsedIntent.CRAWL_RESEARCH: CrawlStrategy.SINGLE_PAGE,
            ParsedIntent.CRAWL_NEWS: CrawlStrategy.RSS_FEED,
            ParsedIntent.CRAWL_BLOG: CrawlStrategy.RSS_FEED,
        }
        return mapping.get(intent, CrawlStrategy.SINGLE_PAGE)
    
    def _determine_content_type(self, text: str) -> ContentType:
        """Determine content type from text."""
        for content_type, indicators in self.content_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    return content_type
        
        return ContentType.GENERAL
    
    def _extract_depth(self, text: str) -> Optional[int]:
        """Extract crawl depth from text."""
        for pattern in self.depth_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Default depths based on keywords
        if 'deep' in text or 'recursive' in text:
            return 3
        elif 'shallow' in text:
            return 1
        
        return None
    
    def _extract_page_limit(self, text: str) -> Optional[int]:
        """Extract page limit from text."""
        for pattern in self.page_limit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Default limits based on intent
        if 'all' in text or 'entire' in text:
            return 100
        elif 'few' in text:
            return 5
        
        return None
    
    def _extract_filters(self, text: str) -> Dict[str, Any]:
        """Extract filtering criteria from text."""
        filters = {}
        
        # Date filters
        if 'today' in text:
            filters['since'] = 'today'
        elif 'yesterday' in text:
            filters['since'] = 'yesterday'
        elif 'this week' in text:
            filters['since'] = 'week'
        elif 'this month' in text:
            filters['since'] = 'month'
        elif match := re.search(r'(?:since|from|after)\s+(\d{4})', text):
            filters['since'] = match.group(1)
        
        # Topic filters
        if match := re.search(r'about\s+([^.!?]+)', text):
            filters['topic'] = match.group(1).strip()
        
        # Author filters
        if match := re.search(r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text):
            filters['author'] = match.group(1)
        
        # Language filters
        if 'in english' in text:
            filters['language'] = 'en'
        elif 'in spanish' in text:
            filters['language'] = 'es'
        elif 'in french' in text:
            filters['language'] = 'fr'
        
        return filters
    
    def _calculate_confidence(
        self,
        urls: List[str],
        intent: ParsedIntent,
        has_limits: bool
    ) -> float:
        """Calculate confidence score for the parse."""
        confidence = 0.5
        
        # URL extraction confidence
        if urls:
            confidence += 0.3
        
        # Intent detection confidence
        if intent != ParsedIntent.UNKNOWN:
            confidence += 0.1
        
        # Specificity bonus
        if has_limits:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def suggest_improvements(self, command: CrawlCommand) -> List[str]:
        """Suggest improvements for low-confidence commands."""
        suggestions = []
        
        if not command.urls:
            suggestions.append("Please specify a URL to crawl (e.g., 'crawl https://example.com')")
        
        if command.confidence < 0.7:
            suggestions.append("Try being more specific about what you want to crawl")
        
        if command.intent == ParsedIntent.UNKNOWN:
            suggestions.append("Specify whether you want a single page or entire site")
        
        if not command.max_pages and command.strategy in [CrawlStrategy.RECURSIVE, CrawlStrategy.SITE_MAP]:
            suggestions.append("Consider adding a page limit (e.g., 'crawl up to 20 pages')")
        
        return suggestions


# Example usage
def example_usage():
    """Example of using the natural language parser."""
    parser = NaturalLanguageCrawlParser()
    
    examples = [
        "Crawl the Wikipedia page about machine learning and save it to memory",
        "Read the entire documentation at docs.python.org up to 50 pages",
        "Get me all blog posts from medium.com about AI written this month",
        "Research arxiv papers on transformer architectures, limit to 10 papers",
        "Crawl https://example.com recursively up to 3 levels deep",
        "Fetch the latest news from techcrunch.com without saving entities",
    ]
    
    for example in examples:
        print(f"\nInput: {example}")
        command = parser.parse(example)
        print(f"Parsed command:")
        print(f"  URLs: {command.urls}")
        print(f"  Intent: {command.intent}")
        print(f"  Strategy: {command.strategy}")
        print(f"  Content type: {command.content_type}")
        print(f"  Max pages: {command.max_pages}")
        print(f"  Depth: {command.depth}")
        print(f"  Filters: {command.filters}")
        print(f"  Confidence: {command.confidence}")
        
        suggestions = parser.suggest_improvements(command)
        if suggestions:
            print(f"  Suggestions: {suggestions}")


if __name__ == "__main__":
    example_usage()