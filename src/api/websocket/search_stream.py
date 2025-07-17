"""
Live Search Streaming for Real-time Memory Streams.

This module provides real-time search capabilities including progressive search,
query suggestions, result refinement, and collaborative search features.
All processing is performed locally with zero external dependencies.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import weakref
import difflib

# Local NLP and search imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.embeddings.embedder import Embedder
from .server import ConnectionManager, MessageType

logger = structlog.get_logger(__name__)


class SearchEventType(str, Enum):
    """Types of search events."""
    QUERY_STARTED = "query_started"
    QUERY_UPDATED = "query_updated"
    RESULTS_FOUND = "results_found"
    RESULTS_REFINED = "results_refined"
    SUGGESTION_GENERATED = "suggestion_generated"
    SEARCH_COMPLETED = "search_completed"
    SEARCH_CANCELLED = "search_cancelled"


class SearchState(str, Enum):
    """States of search sessions."""
    IDLE = "idle"
    SEARCHING = "searching"
    REFINING = "refining"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class QuerySuggestion(BaseModel):
    """Query suggestion with relevance scoring."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str = Field(description="Suggested query text")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    category: str = Field(description="Suggestion category")
    explanation: str = Field(description="Why this suggestion is relevant")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


@dataclass
class SearchEvent:
    """Represents a search-related event."""
    id: str
    event_type: SearchEventType
    search_session_id: str
    user_id: str
    timestamp: datetime
    query: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "search_session_id": self.search_session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "data": self.data,
            "metadata": self.metadata
        }


@dataclass
class SearchSession:
    """Represents an active search session."""
    id: str
    connection_id: str
    user_id: str
    current_query: str
    state: SearchState
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    query_history: List[str] = field(default_factory=list)
    results_count: int = 0
    suggestions_generated: int = 0
    refinements_count: int = 0
    
    def add_query(self, query: str):
        """Add a query to the session history."""
        self.current_query = query
        self.query_history.append(query)
        self.last_activity = datetime.utcnow()


class TrieNode:
    """Node for trie data structure used in query suggestions."""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_word = False
        self.frequency = 0
        self.suggestions: List[str] = []


class QueryTrie:
    """
    Trie data structure for efficient query suggestions.
    
    Features:
    - Fast prefix-based search
    - Frequency-based ranking
    - Auto-completion support
    - Query pattern learning
    """
    
    def __init__(self, max_suggestions: int = 10):
        self.root = TrieNode()
        self.max_suggestions = max_suggestions
        self.total_queries = 0
    
    def insert(self, query: str, frequency: int = 1):
        """Insert a query into the trie."""
        if not query:
            return
        
        node = self.root
        query_lower = query.lower().strip()
        
        # Navigate/create path
        for char in query_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # Mark end of word and update frequency
        node.is_end_word = True
        node.frequency += frequency
        self.total_queries += frequency
        
        # Update suggestions for all prefixes
        self._update_suggestions(query_lower)
    
    def _update_suggestions(self, query: str):
        """Update suggestions for all prefixes of a query."""
        for i in range(1, len(query) + 1):
            prefix = query[:i]
            node = self._find_node(prefix)
            if node:
                suggestions = self._get_completions(node, prefix)
                node.suggestions = suggestions[:self.max_suggestions]
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find the node corresponding to a prefix."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def _get_completions(self, node: TrieNode, prefix: str) -> List[str]:
        """Get all completions from a node, sorted by frequency."""
        completions = []
        
        def dfs(current_node: TrieNode, current_prefix: str):
            if current_node.is_end_word:
                completions.append((current_prefix, current_node.frequency))
            
            for char, child_node in current_node.children.items():
                dfs(child_node, current_prefix + char)
        
        dfs(node, prefix)
        
        # Sort by frequency (descending) and return just the queries
        completions.sort(key=lambda x: x[1], reverse=True)
        return [comp[0] for comp in completions]
    
    def get_suggestions(self, prefix: str) -> List[str]:
        """Get query suggestions for a prefix."""
        if not prefix:
            return []
        
        node = self._find_node(prefix.lower())
        if not node:
            return []
        
        return node.suggestions[:self.max_suggestions]


class SearchStreamManager:
    """
    Manages real-time search streaming and suggestions.
    
    Features:
    - Progressive search with streaming results
    - Real-time query suggestions using trie structures
    - Search result refinement and filtering
    - Collaborative search with shared sessions
    - Query pattern learning and optimization
    - Search analytics and performance tracking
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        embedder: Optional[Embedder] = None,
        redis_cache: Optional[RedisCache] = None,
        suggestion_threshold: float = 0.7,
        max_results_per_batch: int = 20
    ):
        """
        Initialize the search stream manager.
        
        Args:
            connection_manager: WebSocket connection manager
            embedder: Optional embedder for semantic search
            redis_cache: Optional Redis cache for search persistence
            suggestion_threshold: Minimum score for suggestions
            max_results_per_batch: Maximum results per streaming batch
        """
        self.connection_manager = connection_manager
        self.embedder = embedder
        self.redis_cache = redis_cache
        self.suggestion_threshold = suggestion_threshold
        self.max_results_per_batch = max_results_per_batch
        
        # Search sessions management
        self.search_sessions: Dict[str, SearchSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
        
        # Query suggestion system
        self.query_trie = QueryTrie()
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.popular_queries: List[Tuple[str, int]] = []
        
        # TF-IDF for semantic suggestions
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_fitted = False
        self.query_corpus = []
        
        # Search performance tracking
        self.stats = {
            'active_sessions': 0,
            'total_queries': 0,
            'suggestions_generated': 0,
            'results_streamed': 0,
            'average_query_time': 0.0,
            'popular_query_patterns': {}
        }
        
        # Background tasks
        self._suggestion_update_task = None
        self._session_cleanup_task = None
        
        # Register message handlers
        self.connection_manager.register_handler(
            MessageType.SEARCH_QUERY,
            self._handle_search_query
        )
        self.connection_manager.register_handler(
            MessageType.SEARCH_SUGGESTION,
            self._handle_suggestion_request
        )
        
        logger.info(
            "Search stream manager initialized",
            suggestion_threshold=suggestion_threshold,
            max_results_per_batch=max_results_per_batch
        )
    
    async def start(self):
        """Start the search stream manager."""
        self._suggestion_update_task = asyncio.create_task(self._suggestion_update_loop())
        self._session_cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        logger.info("Search stream manager started")
    
    async def stop(self):
        """Stop the search stream manager."""
        if self._suggestion_update_task:
            self._suggestion_update_task.cancel()
        if self._session_cleanup_task:
            self._session_cleanup_task.cancel()
        
        logger.info("Search stream manager stopped")
    
    async def create_search_session(
        self,
        connection_id: str,
        user_id: str,
        initial_query: str = ""
    ) -> str:
        """
        Create a new search session.
        
        Args:
            connection_id: WebSocket connection ID
            user_id: User ID for the session
            initial_query: Optional initial query
            
        Returns:
            Search session ID
        """
        # Generate session ID
        session_id = f"search_{int(time.time() * 1000)}_{connection_id[:8]}"
        
        # Create session
        session = SearchSession(
            id=session_id,
            connection_id=connection_id,
            user_id=user_id,
            current_query=initial_query,
            state=SearchState.IDLE
        )
        
        if initial_query:
            session.add_query(initial_query)
        
        # Store session
        self.search_sessions[session_id] = session
        self.user_sessions[user_id].add(session_id)
        
        # Subscribe to search events
        await self.connection_manager.subscribe(
            connection_id,
            f"search_stream_{session_id}"
        )
        
        # Update statistics
        self.stats['active_sessions'] = len(self.search_sessions)
        
        logger.info(
            "Search session created",
            session_id=session_id,
            user_id=user_id,
            connection_id=connection_id
        )
        
        return session_id
    
    async def end_search_session(self, session_id: str):
        """
        End a search session.
        
        Args:
            session_id: Search session ID to end
        """
        session = self.search_sessions.get(session_id)
        if not session:
            return
        
        # Update session state
        session.state = SearchState.COMPLETED
        
        # Unsubscribe from search events
        await self.connection_manager.unsubscribe(
            session.connection_id,
            f"search_stream_{session_id}"
        )
        
        # Remove session
        del self.search_sessions[session_id]
        self.user_sessions[session.user_id].discard(session_id)
        
        # Clean up empty user session sets
        if not self.user_sessions[session.user_id]:
            del self.user_sessions[session.user_id]
        
        # Learn from session queries
        await self._learn_from_session(session)
        
        # Update statistics
        self.stats['active_sessions'] = len(self.search_sessions)
        
        logger.info(
            "Search session ended",
            session_id=session_id,
            query_count=len(session.query_history)
        )
    
    async def emit_search_event(
        self,
        session_id: str,
        event_type: SearchEventType,
        query: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Emit a search-related event.
        
        Args:
            session_id: Search session ID
            event_type: Type of search event
            query: Search query
            data: Additional event data
        """
        session = self.search_sessions.get(session_id)
        if not session:
            return
        
        # Generate event ID
        event_id = f"search_evt_{int(time.time() * 1000)}_{session_id[:8]}"
        
        # Create event
        event = SearchEvent(
            id=event_id,
            event_type=event_type,
            search_session_id=session_id,
            user_id=session.user_id,
            timestamp=datetime.utcnow(),
            query=query,
            data=data or {}
        )
        
        # Send event to session
        message = {
            "type": MessageType.SEARCH_RESULT.value,
            "event": event.to_dict()
        }
        
        await self.connection_manager.publish_to_subscription(
            f"search_stream_{session_id}",
            message
        )
        
        logger.debug(
            "Search event emitted",
            event_id=event_id,
            event_type=event_type.value,
            session_id=session_id
        )
    
    async def generate_query_suggestions(
        self,
        prefix: str,
        user_id: Optional[str] = None,
        context: Optional[List[str]] = None
    ) -> List[QuerySuggestion]:
        """
        Generate query suggestions for a prefix.
        
        Args:
            prefix: Query prefix to complete
            user_id: Optional user ID for personalization
            context: Optional context for suggestions
            
        Returns:
            List of query suggestions
        """
        suggestions = []
        
        try:
            # Get trie-based suggestions
            trie_suggestions = self.query_trie.get_suggestions(prefix)
            
            # Add trie suggestions
            for i, suggestion in enumerate(trie_suggestions[:5]):
                score = max(0.9 - (i * 0.1), 0.5)  # Decreasing score
                suggestions.append(QuerySuggestion(
                    text=suggestion,
                    score=score,
                    category="completion",
                    explanation="Based on popular queries"
                ))
            
            # Generate semantic suggestions if embedder available
            if self.embedder and len(prefix) > 2:
                semantic_suggestions = await self._generate_semantic_suggestions(prefix)
                suggestions.extend(semantic_suggestions)
            
            # Generate context-based suggestions
            if context:
                context_suggestions = await self._generate_context_suggestions(prefix, context)
                suggestions.extend(context_suggestions)
            
            # Generate pattern-based suggestions
            pattern_suggestions = await self._generate_pattern_suggestions(prefix)
            suggestions.extend(pattern_suggestions)
            
            # Sort by score and remove duplicates
            seen_texts = set()
            unique_suggestions = []
            for suggestion in sorted(suggestions, key=lambda x: x.score, reverse=True):
                if suggestion.text not in seen_texts and suggestion.score >= self.suggestion_threshold:
                    seen_texts.add(suggestion.text)
                    unique_suggestions.append(suggestion)
            
            # Update statistics
            self.stats['suggestions_generated'] += len(unique_suggestions)
            
            return unique_suggestions[:10]  # Return top 10
            
        except Exception as e:
            logger.warning(f"Error generating query suggestions: {e}")
            return []
    
    async def _generate_semantic_suggestions(self, prefix: str) -> List[QuerySuggestion]:
        """Generate semantically similar query suggestions."""
        if not self.tfidf_fitted or not self.query_corpus:
            return []
        
        try:
            # Find semantically similar queries
            prefix_vector = self.tfidf_vectorizer.transform([prefix])
            query_vectors = self.tfidf_vectorizer.transform(self.query_corpus)
            
            similarities = cosine_similarity(prefix_vector, query_vectors)[0]
            
            # Get top similar queries
            similar_indices = np.argsort(similarities)[-5:][::-1]
            
            suggestions = []
            for idx in similar_indices:
                if similarities[idx] > 0.3:  # Minimum similarity threshold
                    query = self.query_corpus[idx]
                    if not query.startswith(prefix.lower()):
                        suggestions.append(QuerySuggestion(
                            text=query,
                            score=float(similarities[idx]),
                            category="semantic",
                            explanation="Semantically similar to your query"
                        ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Error generating semantic suggestions: {e}")
            return []
    
    async def _generate_context_suggestions(
        self, 
        prefix: str, 
        context: List[str]
    ) -> List[QuerySuggestion]:
        """Generate context-based query suggestions."""
        suggestions = []
        
        try:
            # Extract keywords from context
            context_text = " ".join(context).lower()
            context_words = set(context_text.split())
            
            # Find queries that include context words
            for query, frequency in self.query_patterns.items():
                query_words = set(query.split())
                overlap = len(context_words.intersection(query_words))
                
                if overlap > 0 and query.startswith(prefix.lower()):
                    score = min(0.8, overlap * 0.2 + 0.4)
                    suggestions.append(QuerySuggestion(
                        text=query,
                        score=score,
                        category="contextual",
                        explanation=f"Related to context ({overlap} matching terms)"
                    ))
            
            return suggestions[:3]  # Return top 3 context suggestions
            
        except Exception as e:
            logger.warning(f"Error generating context suggestions: {e}")
            return []
    
    async def _generate_pattern_suggestions(self, prefix: str) -> List[QuerySuggestion]:
        """Generate pattern-based query suggestions."""
        suggestions = []
        
        try:
            # Common query patterns
            patterns = [
                f"how to {prefix}",
                f"what is {prefix}",
                f"where is {prefix}",
                f"when was {prefix}",
                f"why does {prefix}",
                f"{prefix} tutorial",
                f"{prefix} examples",
                f"{prefix} best practices"
            ]
            
            for pattern in patterns:
                if len(pattern) > len(prefix) + 3:  # Ensure meaningful extension
                    suggestions.append(QuerySuggestion(
                        text=pattern,
                        score=0.6,
                        category="pattern",
                        explanation="Common query pattern"
                    ))
            
            return suggestions[:2]  # Return top 2 pattern suggestions
            
        except Exception as e:
            logger.warning(f"Error generating pattern suggestions: {e}")
            return []
    
    async def _handle_search_query(self, connection_id: str, data: Dict[str, Any]):
        """Handle search query from WebSocket."""
        try:
            query = data.get('query', '').strip()
            session_id = data.get('session_id')
            
            if not query:
                await self.connection_manager.send_message(connection_id, {
                    "type": MessageType.ERROR.value,
                    "error": "Empty query"
                })
                return
            
            # Get or create session
            if not session_id:
                connection = self.connection_manager.connections.get(connection_id)
                if not connection or not connection.user_id:
                    await self.connection_manager.send_message(connection_id, {
                        "type": MessageType.ERROR.value,
                        "error": "Authentication required"
                    })
                    return
                
                session_id = await self.create_search_session(
                    connection_id,
                    connection.user_id,
                    query
                )
            
            session = self.search_sessions.get(session_id)
            if not session:
                await self.connection_manager.send_message(connection_id, {
                    "type": MessageType.ERROR.value,
                    "error": "Invalid session"
                })
                return
            
            # Update session
            session.add_query(query)
            session.state = SearchState.SEARCHING
            
            # Emit search started event
            await self.emit_search_event(
                session_id,
                SearchEventType.QUERY_STARTED,
                query,
                {"session_id": session_id}
            )
            
            # Learn from query
            self.query_trie.insert(query.lower())
            self.query_patterns[query.lower()] += 1
            self.stats['total_queries'] += 1
            
            # Send acknowledgment
            await self.connection_manager.send_message(connection_id, {
                "type": MessageType.SEARCH_RESULT.value,
                "session_id": session_id,
                "query": query,
                "status": "searching"
            })
            
        except Exception as e:
            logger.warning(f"Error handling search query: {e}")
            await self.connection_manager.send_message(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Search query failed"
            })
    
    async def _handle_suggestion_request(self, connection_id: str, data: Dict[str, Any]):
        """Handle suggestion request from WebSocket."""
        try:
            prefix = data.get('prefix', '').strip()
            context = data.get('context', [])
            
            # Get user ID for personalization
            connection = self.connection_manager.connections.get(connection_id)
            user_id = connection.user_id if connection else None
            
            # Generate suggestions
            suggestions = await self.generate_query_suggestions(prefix, user_id, context)
            
            # Send suggestions
            await self.connection_manager.send_message(connection_id, {
                "type": MessageType.SEARCH_SUGGESTION.value,
                "prefix": prefix,
                "suggestions": [
                    {
                        "text": s.text,
                        "score": s.score,
                        "category": s.category,
                        "explanation": s.explanation
                    }
                    for s in suggestions
                ]
            })
            
        except Exception as e:
            logger.warning(f"Error handling suggestion request: {e}")
            await self.connection_manager.send_message(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Suggestion generation failed"
            })
    
    async def _learn_from_session(self, session: SearchSession):
        """Learn patterns from a completed search session."""
        try:
            # Add queries to corpus for semantic suggestions
            for query in session.query_history:
                if query not in self.query_corpus:
                    self.query_corpus.append(query.lower())
            
            # Refit TF-IDF if we have enough queries
            if len(self.query_corpus) >= 10:
                try:
                    self.tfidf_vectorizer.fit(self.query_corpus)
                    self.tfidf_fitted = True
                except:
                    pass  # TF-IDF fitting can fail with small datasets
            
            # Update popular queries
            query_counter = Counter(self.query_patterns)
            self.popular_queries = query_counter.most_common(100)
            
            # Update statistics
            self.stats['popular_query_patterns'] = dict(query_counter.most_common(10))
            
        except Exception as e:
            logger.warning(f"Error learning from session: {e}")
    
    async def _suggestion_update_loop(self):
        """Background task for updating suggestion models."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Update popular queries
                if self.query_patterns:
                    query_counter = Counter(self.query_patterns)
                    self.popular_queries = query_counter.most_common(100)
                    self.stats['popular_query_patterns'] = dict(query_counter.most_common(10))
                
                # Maintain corpus size
                if len(self.query_corpus) > 10000:
                    self.query_corpus = self.query_corpus[-5000:]  # Keep recent 5000
                    if self.query_corpus:
                        try:
                            self.tfidf_vectorizer.fit(self.query_corpus)
                        except:
                            pass
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in suggestion update loop: {e}")
    
    async def _session_cleanup_loop(self):
        """Background task for cleaning up stale search sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.utcnow()
                stale_sessions = []
                
                # Find stale sessions (no activity for 30 minutes)
                for session_id, session in self.search_sessions.items():
                    if (current_time - session.last_activity).total_seconds() > 1800:
                        stale_sessions.append(session_id)
                
                # Clean up stale sessions
                for session_id in stale_sessions:
                    await self.end_search_session(session_id)
                
                if stale_sessions:
                    logger.info(f"Cleaned up {len(stale_sessions)} stale search sessions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in session cleanup loop: {e}")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search stream statistics."""
        return {
            **self.stats,
            'query_corpus_size': len(self.query_corpus),
            'unique_query_patterns': len(self.query_patterns),
            'tfidf_fitted': self.tfidf_fitted,
            'active_sessions_by_state': {
                state.value: len([
                    s for s in self.search_sessions.values() 
                    if s.state == state
                ])
                for state in SearchState
            }
        }
    
    async def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired or inactive search sessions."""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, session in self.search_sessions.items():
                age_hours = (current_time - session.last_activity).total_seconds() / 3600
                if age_hours > max_age_hours:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.search_sessions[session_id]
                logger.info(
                    "Removed expired search session",
                    session_id=session_id,
                    age_hours=age_hours
                )
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error("Error cleaning up expired search sessions", error=str(e))
            return 0
    
    def optimize_suggestion_trie(self):
        """Optimize the suggestion trie for better performance."""
        try:
            # Rebuild trie with frequency-based ordering
            all_queries = []
            for session in self.search_sessions.values():
                for query in session.query_history:
                    all_queries.append(query.lower().strip())
            
            # Count query frequencies
            query_counts = {}
            for query in all_queries:
                query_counts[query] = query_counts.get(query, 0) + 1
            
            # Rebuild suggestion trie with popular queries first
            self.suggestion_trie = {}
            sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
            
            for query, count in sorted_queries:
                self._add_to_trie(query, count)
            
            logger.info(
                "Optimized suggestion trie",
                total_queries=len(sorted_queries),
                unique_queries=len(query_counts)
            )
            
        except Exception as e:
            logger.error("Error optimizing suggestion trie", error=str(e))
    
    def _add_to_trie(self, query: str, frequency: int):
        """Add a query to the suggestion trie with frequency weighting."""
        node = self.suggestion_trie
        for char in query:
            if char not in node:
                node[char] = {"_children": {}, "_frequency": 0, "_completions": []}
            node = node[char]
            node["_frequency"] += frequency
            
            # Keep top completions based on frequency
            if query not in [c["query"] for c in node["_completions"]]:
                node["_completions"].append({"query": query, "frequency": frequency})
                node["_completions"] = sorted(
                    node["_completions"], 
                    key=lambda x: x["frequency"], 
                    reverse=True
                )[:10]  # Keep top 10 completions


# Export main components
__all__ = [
    "SearchStreamManager",
    "SearchEvent",
    "SearchEventType", 
    "QuerySuggestion",
    "SearchState",
    "SearchSession",
    "SearchQuery"
]