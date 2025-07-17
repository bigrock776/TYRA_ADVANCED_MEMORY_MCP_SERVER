"""
Neo4j Graph Linker for Tyra Web Memory System.

Advanced Neo4j integration with Graphiti-compatible schema for storing and managing
relationships between web sources, entities, topics, and memory chunks. Provides
comprehensive graph operations for knowledge mapping and trust scoring.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import uuid
from urllib.parse import urlparse

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, TransientError
from pydantic import BaseModel, Field, ConfigDict

from ..agents.websearch_agent import WebSearchResult
from ..core.utils.config import settings

logger = structlog.get_logger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    PAGE = "Page"
    ENTITY = "Entity"
    TOPIC = "Topic"
    SOURCE = "Source"
    MEMORY_CHUNK = "MemoryChunk"
    QUERY = "Query"
    DOMAIN = "Domain"


class RelationType(str, Enum):
    """Types of relationships in the knowledge graph."""
    MENTIONS = "MENTIONS"
    DERIVED_FROM = "DERIVED_FROM"
    TRUSTS = "TRUSTS"
    SIMILAR_TO = "SIMILAR_TO"
    CONTAINS = "CONTAINS"
    LINKS_TO = "LINKS_TO"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    ANSWERED_BY = "ANSWERED_BY"
    HOSTED_BY = "HOSTED_BY"
    REFERENCES = "REFERENCES"


class TrustLevel(str, Enum):
    """Trust levels for sources and content."""
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNTRUSTED = "untrusted"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: NodeType
    properties: Dict[str, Any]
    labels: List[str]


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""
    start_node_id: str
    end_node_id: str
    type: RelationType
    properties: Dict[str, Any]


class Neo4jLinker:
    """
    Advanced Neo4j Graph Linker for Tyra Web Memory System.
    
    Provides comprehensive graph operations for storing and managing relationships
    between web sources, entities, topics, and memory chunks with Graphiti compatibility.
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50
    ):
        """
        Initialize Neo4j Graph Linker.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
            max_connection_lifetime: Maximum connection lifetime in seconds
            max_connection_pool_size: Maximum connection pool size
        """
        self.uri = uri or self._build_neo4j_uri()
        self.user = user or settings.memory.neo4j.user
        self.password = password or settings.memory.neo4j.password
        self.database = database
        
        # Connection configuration
        self.driver: Optional[AsyncDriver] = None
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        
        # Performance tracking
        self.operation_stats = {
            'total_queries': 0,
            'total_writes': 0,
            'total_reads': 0,
            'average_query_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'connection_errors': 0
        }
        
        # Domain trust scores cache
        self.domain_trust_cache: Dict[str, float] = {}
        
        logger.info(
            "Initialized Neo4jLinker",
            uri=self.uri,
            database=database
        )
    
    def _build_neo4j_uri(self) -> str:
        """Build Neo4j URI from settings."""
        neo4j_config = settings.memory.neo4j
        return f"bolt://{neo4j_config.host}:{neo4j_config.port}"
    
    async def initialize(self) -> None:
        """Initialize Neo4j connection and schema."""
        try:
            # Create driver
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size
            )
            
            # Verify connectivity
            await self.driver.verify_connectivity()
            
            # Create constraints and indexes
            await self._create_schema()
            
            logger.info("Neo4j initialization completed successfully")
            
        except Exception as e:
            logger.error("Neo4j initialization failed", error=str(e))
            raise
    
    async def _create_schema(self) -> None:
        """Create Neo4j schema with constraints and indexes."""
        schema_queries = [
            # Constraints for uniqueness
            "CREATE CONSTRAINT page_id_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT memory_chunk_id_unique IF NOT EXISTS FOR (m:MemoryChunk) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT query_id_unique IF NOT EXISTS FOR (q:Query) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT domain_id_unique IF NOT EXISTS FOR (d:Domain) REQUIRE d.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX page_url_index IF NOT EXISTS FOR (p:Page) ON (p.url)",
            "CREATE INDEX page_timestamp_index IF NOT EXISTS FOR (p:Page) ON (p.timestamp)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX topic_name_index IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX source_domain_index IF NOT EXISTS FOR (s:Source) ON (s.domain)",
            "CREATE INDEX memory_chunk_text_index IF NOT EXISTS FOR (m:MemoryChunk) ON (m.text)",
            "CREATE INDEX query_text_index IF NOT EXISTS FOR (q:Query) ON (q.text)",
            "CREATE INDEX domain_name_index IF NOT EXISTS FOR (d:Domain) ON (d.name)",
            
            # Trust and confidence indexes
            "CREATE INDEX source_trust_index IF NOT EXISTS FOR (s:Source) ON (s.trust_score)",
            "CREATE INDEX page_confidence_index IF NOT EXISTS FOR (p:Page) ON (p.confidence_score)",
            "CREATE INDEX memory_confidence_index IF NOT EXISTS FOR (m:MemoryChunk) ON (m.confidence_score)",
        ]
        
        async with self.driver.session(database=self.database) as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                    logger.debug("Created schema element", query=query)
                except Exception as e:
                    # Constraint might already exist
                    logger.debug("Schema creation skipped", query=query, reason=str(e))
    
    async def add_web_search_result(
        self,
        result: WebSearchResult,
        query: str,
        additional_entities: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Add web search result to knowledge graph.
        
        Args:
            result: Web search result to add
            query: Original search query
            additional_entities: Additional entities to extract
            
        Returns:
            Dictionary with created node IDs
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.driver.session(database=self.database) as session:
                # Create transaction for atomicity
                async with session.begin_transaction() as tx:
                    created_nodes = {}
                    
                    # Parse URL for domain information
                    parsed_url = urlparse(result.source)
                    domain = parsed_url.netloc
                    
                    # Create or update domain node
                    domain_id = await self._create_domain_node(tx, domain)
                    created_nodes['domain'] = domain_id
                    
                    # Create or update source node
                    source_id = await self._create_source_node(tx, result.source, domain_id)
                    created_nodes['source'] = source_id
                    
                    # Create page node
                    page_id = await self._create_page_node(tx, result, source_id)
                    created_nodes['page'] = page_id
                    
                    # Create memory chunk node
                    chunk_id = await self._create_memory_chunk_node(tx, result, page_id)
                    created_nodes['memory_chunk'] = chunk_id
                    
                    # Create query node
                    query_id = await self._create_query_node(tx, query)
                    created_nodes['query'] = query_id
                    
                    # Extract and create entities
                    entities = await self._extract_entities(result.text, additional_entities)
                    entity_ids = []
                    for entity in entities:
                        entity_id = await self._create_entity_node(tx, entity, page_id)
                        entity_ids.append(entity_id)
                    created_nodes['entities'] = entity_ids
                    
                    # Extract and create topics
                    topics = await self._extract_topics(result.text)
                    topic_ids = []
                    for topic in topics:
                        topic_id = await self._create_topic_node(tx, topic, page_id)
                        topic_ids.append(topic_id)
                    created_nodes['topics'] = topic_ids
                    
                    # Create relationships
                    await self._create_relationships(tx, created_nodes, result)
                    
                    # Update trust scores
                    await self._update_trust_scores(tx, domain_id, source_id, result.confidence_score)
                    
                await tx.commit()
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.operation_stats['total_writes'] += 1
            self.operation_stats['average_query_time'] = (
                (self.operation_stats['average_query_time'] * (self.operation_stats['total_queries']) + 
                 query_time) / (self.operation_stats['total_queries'] + 1)
            )
            self.operation_stats['total_queries'] += 1
            
            logger.info(
                "Added web search result to graph",
                source=result.source,
                title=result.title,
                entities_count=len(entity_ids),
                topics_count=len(topic_ids),
                query_time_seconds=query_time
            )
            
            return created_nodes
            
        except Exception as e:
            logger.error("Failed to add web search result to graph", error=str(e))
            self.operation_stats['connection_errors'] += 1
            raise
    
    async def _create_domain_node(self, tx, domain: str) -> str:
        """Create or update domain node."""
        domain_id = f"domain:{hashlib.md5(domain.encode()).hexdigest()[:8]}"
        
        query = """
        MERGE (d:Domain {id: $domain_id})
        ON CREATE SET
            d.name = $domain,
            d.created_at = datetime(),
            d.trust_score = 0.5,
            d.page_count = 0,
            d.last_seen = datetime()
        ON MATCH SET
            d.last_seen = datetime(),
            d.page_count = d.page_count + 1
        RETURN d.id AS id
        """
        
        result = await tx.run(query, domain_id=domain_id, domain=domain)
        record = await result.single()
        return record['id']
    
    async def _create_source_node(self, tx, url: str, domain_id: str) -> str:
        """Create or update source node."""
        source_id = f"source:{hashlib.md5(url.encode()).hexdigest()[:12]}"
        parsed_url = urlparse(url)
        
        query = """
        MERGE (s:Source {id: $source_id})
        ON CREATE SET
            s.url = $url,
            s.domain = $domain,
            s.scheme = $scheme,
            s.path = $path,
            s.created_at = datetime(),
            s.trust_score = 0.5,
            s.access_count = 1,
            s.last_accessed = datetime()
        ON MATCH SET
            s.access_count = s.access_count + 1,
            s.last_accessed = datetime()
        WITH s
        MATCH (d:Domain {id: $domain_id})
        MERGE (s)-[:HOSTED_BY]->(d)
        RETURN s.id AS id
        """
        
        result = await tx.run(
            query,
            source_id=source_id,
            url=url,
            domain=parsed_url.netloc,
            scheme=parsed_url.scheme,
            path=parsed_url.path,
            domain_id=domain_id
        )
        record = await result.single()
        return record['id']
    
    async def _create_page_node(self, tx, result: WebSearchResult, source_id: str) -> str:
        """Create page node."""
        page_id = f"page:{hashlib.md5(result.source.encode()).hexdigest()[:12]}:{int(result.timestamp.timestamp())}"
        
        query = """
        CREATE (p:Page {
            id: $page_id,
            url: $url,
            title: $title,
            text_length: $text_length,
            extraction_method: $extraction_method,
            extraction_quality: $extraction_quality,
            confidence_score: $confidence_score,
            freshness_score: $freshness_score,
            relevance_score: $relevance_score,
            language: $language,
            timestamp: datetime($timestamp),
            created_at: datetime()
        })
        WITH p
        MATCH (s:Source {id: $source_id})
        MERGE (p)-[:EXTRACTED_FROM]->(s)
        RETURN p.id AS id
        """
        
        result_data = await tx.run(
            query,
            page_id=page_id,
            url=result.source,
            title=result.title,
            text_length=result.processed_length,
            extraction_method=result.extraction_method.value,
            extraction_quality=result.extraction_quality.value,
            confidence_score=result.confidence_score,
            freshness_score=result.freshness_score,
            relevance_score=result.relevance_score,
            language=result.language,
            timestamp=result.timestamp.isoformat(),
            source_id=source_id
        )
        record = await result_data.single()
        return record['id']
    
    async def _create_memory_chunk_node(self, tx, result: WebSearchResult, page_id: str) -> str:
        """Create memory chunk node."""
        chunk_id = f"chunk:{hashlib.md5(result.text.encode()).hexdigest()[:16]}"
        
        query = """
        MERGE (m:MemoryChunk {id: $chunk_id})
        ON CREATE SET
            m.text = $text,
            m.text_hash = $text_hash,
            m.summary = $summary,
            m.confidence_score = $confidence_score,
            m.relevance_score = $relevance_score,
            m.freshness_score = $freshness_score,
            m.length = $length,
            m.created_at = datetime(),
            m.last_accessed = datetime(),
            m.access_count = 1
        ON MATCH SET
            m.last_accessed = datetime(),
            m.access_count = m.access_count + 1
        WITH m
        MATCH (p:Page {id: $page_id})
        MERGE (m)-[:DERIVED_FROM]->(p)
        RETURN m.id AS id
        """
        
        text_hash = hashlib.sha256(result.text.encode()).hexdigest()
        
        result_data = await tx.run(
            query,
            chunk_id=chunk_id,
            text=result.text[:1000],  # Truncate for storage
            text_hash=text_hash,
            summary=result.summary,
            confidence_score=result.confidence_score,
            relevance_score=result.relevance_score,
            freshness_score=result.freshness_score,
            length=len(result.text),
            page_id=page_id
        )
        record = await result_data.single()
        return record['id']
    
    async def _create_query_node(self, tx, query_text: str) -> str:
        """Create or update query node."""
        query_id = f"query:{hashlib.md5(query_text.encode()).hexdigest()[:12]}"
        
        query = """
        MERGE (q:Query {id: $query_id})
        ON CREATE SET
            q.text = $query_text,
            q.created_at = datetime(),
            q.use_count = 1,
            q.last_used = datetime()
        ON MATCH SET
            q.use_count = q.use_count + 1,
            q.last_used = datetime()
        RETURN q.id AS id
        """
        
        result = await tx.run(query, query_id=query_id, query_text=query_text)
        record = await result.single()
        return record['id']
    
    async def _extract_entities(self, text: str, additional_entities: Optional[List[str]] = None) -> List[str]:
        """Extract entities from text using simple heuristics."""
        entities = set()
        
        # Simple entity extraction (could be enhanced with spaCy NER)
        words = text.split()
        
        # Look for capitalized words (potential entities)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if it's not at sentence start
                if i > 0 and words[i-1][-1] not in '.!?':
                    entities.add(word.strip('.,!?;:'))
        
        # Add additional entities if provided
        if additional_entities:
            entities.update(additional_entities)
        
        return list(entities)[:10]  # Limit to 10 entities
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword analysis."""
        # Simple topic extraction (could be enhanced with topic modeling)
        keywords = []
        
        # Common technical and business terms
        topic_patterns = [
            'artificial intelligence', 'machine learning', 'deep learning',
            'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum',
            'data science', 'analytics', 'database', 'cloud computing',
            'software development', 'programming', 'API', 'algorithm',
            'security', 'privacy', 'encryption', 'authentication'
        ]
        
        text_lower = text.lower()
        for pattern in topic_patterns:
            if pattern in text_lower:
                keywords.append(pattern)
        
        return keywords[:5]  # Limit to 5 topics
    
    async def _create_entity_node(self, tx, entity: str, page_id: str) -> str:
        """Create entity node."""
        entity_id = f"entity:{hashlib.md5(entity.lower().encode()).hexdigest()[:12]}"
        
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET
            e.name = $entity,
            e.normalized_name = $normalized_name,
            e.created_at = datetime(),
            e.mention_count = 1,
            e.last_mentioned = datetime()
        ON MATCH SET
            e.mention_count = e.mention_count + 1,
            e.last_mentioned = datetime()
        WITH e
        MATCH (p:Page {id: $page_id})
        MERGE (p)-[:MENTIONS]->(e)
        RETURN e.id AS id
        """
        
        result = await tx.run(
            query,
            entity_id=entity_id,
            entity=entity,
            normalized_name=entity.lower().strip(),
            page_id=page_id
        )
        record = await result.single()
        return record['id']
    
    async def _create_topic_node(self, tx, topic: str, page_id: str) -> str:
        """Create topic node."""
        topic_id = f"topic:{hashlib.md5(topic.lower().encode()).hexdigest()[:12]}"
        
        query = """
        MERGE (t:Topic {id: $topic_id})
        ON CREATE SET
            t.name = $topic,
            t.normalized_name = $normalized_name,
            t.created_at = datetime(),
            t.mention_count = 1,
            t.last_mentioned = datetime()
        ON MATCH SET
            t.mention_count = t.mention_count + 1,
            t.last_mentioned = datetime()
        WITH t
        MATCH (p:Page {id: $page_id})
        MERGE (p)-[:CONTAINS]->(t)
        RETURN t.id AS id
        """
        
        result = await tx.run(
            query,
            topic_id=topic_id,
            topic=topic,
            normalized_name=topic.lower().strip(),
            page_id=page_id
        )
        record = await result.single()
        return record['id']
    
    async def _create_relationships(self, tx, created_nodes: Dict[str, Any], result: WebSearchResult) -> None:
        """Create additional relationships between nodes."""
        query_id = created_nodes['query']
        page_id = created_nodes['page']
        chunk_id = created_nodes['memory_chunk']
        
        # Query answered by page
        await tx.run(
            "MATCH (q:Query {id: $query_id}), (p:Page {id: $page_id}) "
            "MERGE (q)-[:ANSWERED_BY {confidence: $confidence, timestamp: datetime()}]->(p)",
            query_id=query_id, page_id=page_id, confidence=result.confidence_score
        )
        
        # Memory chunk references page
        await tx.run(
            "MATCH (m:MemoryChunk {id: $chunk_id}), (p:Page {id: $page_id}) "
            "MERGE (m)-[:REFERENCES {relevance: $relevance, timestamp: datetime()}]->(p)",
            chunk_id=chunk_id, page_id=page_id, relevance=result.relevance_score
        )
    
    async def _update_trust_scores(self, tx, domain_id: str, source_id: str, confidence: float) -> None:
        """Update trust scores based on content confidence."""
        # Update domain trust score (moving average)
        await tx.run(
            """
            MATCH (d:Domain {id: $domain_id})
            SET d.trust_score = (d.trust_score * 0.9) + ($confidence * 0.1)
            """,
            domain_id=domain_id, confidence=confidence
        )
        
        # Update source trust score
        await tx.run(
            """
            MATCH (s:Source {id: $source_id})
            SET s.trust_score = (s.trust_score * 0.8) + ($confidence * 0.2)
            """,
            source_id=source_id, confidence=confidence
        )
    
    async def find_similar_content(
        self,
        text: str,
        similarity_threshold: float = 0.75,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar content in the knowledge graph."""
        try:
            async with self.driver.session(database=self.database) as session:
                # Simple text-based similarity (could be enhanced with embeddings)
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                
                query = """
                MATCH (m:MemoryChunk)
                WHERE m.text_hash <> $text_hash
                WITH m, size(split(m.text, ' ')) as chunk_words, 
                     size(split($text, ' ')) as query_words
                WHERE abs(chunk_words - query_words) / toFloat(max(chunk_words, query_words)) < 0.5
                RETURN m.id as id, m.text as text, m.confidence_score as confidence,
                       m.created_at as created_at
                ORDER BY m.confidence_score DESC
                LIMIT $limit
                """
                
                result = await session.run(query, text_hash=text_hash, text=text[:1000], limit=limit)
                records = await result.data()
                
                self.operation_stats['total_reads'] += 1
                
                return records
                
        except Exception as e:
            logger.error("Failed to find similar content", error=str(e))
            return []
    
    async def get_domain_trust_score(self, domain: str) -> float:
        """Get trust score for a domain."""
        if domain in self.domain_trust_cache:
            return self.domain_trust_cache[domain]
        
        try:
            async with self.driver.session(database=self.database) as session:
                query = """
                MATCH (d:Domain {name: $domain})
                RETURN d.trust_score as trust_score
                """
                
                result = await session.run(query, domain=domain)
                record = await result.single()
                
                if record:
                    trust_score = record['trust_score']
                    self.domain_trust_cache[domain] = trust_score
                    return trust_score
                else:
                    # Default trust score for unknown domains
                    default_score = 0.5
                    self.domain_trust_cache[domain] = default_score
                    return default_score
                    
        except Exception as e:
            logger.error("Failed to get domain trust score", domain=domain, error=str(e))
            return 0.5
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        try:
            async with self.driver.session(database=self.database) as session:
                # Count nodes by type
                node_counts_query = """
                CALL apoc.meta.stats() YIELD labels
                RETURN labels
                """
                
                # Simplified stats query for systems without APOC
                stats_query = """
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                """
                
                try:
                    result = await session.run(node_counts_query)
                    apoc_stats = await result.single()
                    if apoc_stats:
                        node_stats = apoc_stats['labels']
                    else:
                        raise Exception("APOC not available")
                except:
                    # Fallback to manual counting
                    result = await session.run(stats_query)
                    records = await result.data()
                    node_stats = {}
                    for record in records:
                        for label in record['labels']:
                            if label not in node_stats:
                                node_stats[label] = 0
                            node_stats[label] += record['count']
                
                # Count relationships
                rel_query = """
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                """
                
                result = await session.run(rel_query)
                rel_records = await result.data()
                relationship_stats = {record['relationship_type']: record['count'] for record in rel_records}
                
                return {
                    'node_counts': node_stats,
                    'relationship_counts': relationship_stats,
                    'total_nodes': sum(node_stats.values()),
                    'total_relationships': sum(relationship_stats.values()),
                    'operation_stats': self.operation_stats
                }
                
        except Exception as e:
            logger.error("Failed to get graph statistics", error=str(e))
            return {'error': str(e)}
    
    async def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """Clean up old data from the graph."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.driver.session(database=self.database) as session:
                # Delete old pages and their relationships
                cleanup_query = """
                MATCH (p:Page)
                WHERE p.created_at < datetime($cutoff_date)
                OPTIONAL MATCH (p)-[r1]-()
                OPTIONAL MATCH ()-[r2]-(p)
                DELETE r1, r2, p
                RETURN count(p) as deleted_pages
                """
                
                result = await session.run(cleanup_query, cutoff_date=cutoff_date.isoformat())
                record = await result.single()
                deleted_pages = record['deleted_pages'] if record else 0
                
                logger.info(
                    "Cleanup completed",
                    deleted_pages=deleted_pages,
                    cutoff_days=days
                )
                
                return {
                    'deleted_pages': deleted_pages,
                    'cutoff_days': days
                }
                
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of Neo4j connection."""
        try:
            async with self.driver.session(database=self.database) as session:
                # Test basic connectivity
                result = await session.run("RETURN 1 as test")
                test_result = await result.single()
                
                if test_result and test_result['test'] == 1:
                    # Get basic database info
                    stats = await self.get_graph_stats()
                    
                    return {
                        'status': 'healthy',
                        'connected': True,
                        'database': self.database,
                        'node_count': stats.get('total_nodes', 0),
                        'relationship_count': stats.get('total_relationships', 0),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'connected': False,
                        'error': 'Test query failed',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def close(self) -> None:
        """Close Neo4j connections."""
        if self.driver:
            await self.driver.close()
        logger.info("Neo4j connections closed")


# Example usage
async def example_usage():
    """Example of using Neo4jLinker."""
    linker = Neo4jLinker()
    
    try:
        # Initialize
        await linker.initialize()
        
        # Mock web search result for testing
        from ..agents.websearch_agent import WebSearchResult, ContentExtractor, ExtractionQuality
        
        result = WebSearchResult(
            text="This is a test article about artificial intelligence and machine learning.",
            title="AI Test Article",
            source="https://example.com/ai-article",
            embedding=[0.1] * 384,
            extraction_method=ContentExtractor.TRAFILATURA,
            extraction_quality=ExtractionQuality.GOOD,
            confidence_score=0.85,
            freshness_score=0.9,
            relevance_score=0.8,
            content_length=1000,
            processed_length=500
        )
        
        # Add to graph
        nodes = await linker.add_web_search_result(result, "artificial intelligence")
        print(f"Created nodes: {nodes}")
        
        # Get statistics
        stats = await linker.get_graph_stats()
        print(f"Graph stats: {stats}")
        
        # Health check
        health = await linker.health_check()
        print(f"Health status: {health}")
        
    finally:
        await linker.close()


if __name__ == "__main__":
    asyncio.run(example_usage())