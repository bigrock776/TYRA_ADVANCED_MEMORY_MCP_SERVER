"""
Contextual Chunk Linking System for Advanced RAG Pipeline.

This module provides intelligent chunk relationship analysis and contextual linking
to enhance retrieval accuracy through semantic connections between text chunks.
All processing is performed locally with zero external API calls.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import heapq
from collections import defaultdict, deque
import json
import hashlib

# Scientific computing imports
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import pandas as pd

# NLP imports
import spacy
from sentence_transformers import SentenceTransformer
import re

import structlog
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent

from ...models.memory import Memory
from ..embeddings.embedder import Embedder
from ..cache.redis_cache import RedisCache
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class LinkType(str, Enum):
    """Types of contextual links between chunks."""
    SEQUENTIAL = "sequential"          # Natural text sequence
    SEMANTIC = "semantic"              # Conceptual similarity
    CAUSAL = "causal"                 # Cause-effect relationship
    TEMPORAL = "temporal"              # Time-based relationship
    REFERENCE = "reference"            # Cross-reference/citation
    HIERARCHICAL = "hierarchical"      # Parent-child structure
    ELABORATIVE = "elaborative"        # Expansion/detail relationship
    CONTRADICTORY = "contradictory"    # Conflicting information
    COMPLEMENTARY = "complementary"    # Supporting information


class ChunkType(str, Enum):
    """Types of content chunks."""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SECTION = "section"
    DOCUMENT = "document"
    CODE_BLOCK = "code_block"
    LIST_ITEM = "list_item"
    TITLE = "title"
    ABSTRACT = "abstract"


class LinkStrength(str, Enum):
    """Strength of contextual links."""
    VERY_STRONG = "very_strong"    # 0.9+
    STRONG = "strong"              # 0.7-0.9
    MODERATE = "moderate"          # 0.5-0.7
    WEAK = "weak"                  # 0.3-0.5
    VERY_WEAK = "very_weak"        # 0.1-0.3


@dataclass
class TextChunk:
    """Structured representation of a text chunk."""
    id: str
    content: str
    chunk_type: ChunkType
    start_position: int
    end_position: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    source_memory_id: str = ""


@dataclass
class ContextualLink:
    """Represents a contextual relationship between chunks."""
    id: str
    source_chunk_id: str
    target_chunk_id: str
    link_type: LinkType
    strength: LinkStrength
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ChunkLinkingResult(BaseModel):
    """Result from chunk linking analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chunks: List[TextChunk] = Field(description="Extracted text chunks")
    links: List[ContextualLink] = Field(description="Contextual links between chunks")
    graph_metrics: Dict[str, float] = Field(description="Graph analysis metrics")
    processing_time: float = Field(description="Processing time in seconds")


class ContextualChunkLinker:
    """
    Advanced contextual chunk linking system for RAG pipeline.
    
    Features:
    - Intelligent text chunking with context preservation
    - Multi-type relationship detection (semantic, temporal, causal, etc.)
    - Graph-based link analysis and traversal
    - Adaptive chunk sizing based on content structure
    - Entity and keyword-based relationship inference
    - Local NLP processing with spaCy and sentence-transformers
    """
    
    def __init__(
        self,
        embedder: Embedder,
        cache: Optional[RedisCache] = None,
        spacy_model: str = "en_core_web_sm",
        similarity_threshold: float = 0.75,
        max_chunk_size: int = 512,
        overlap_size: int = 50
    ):
        """
        Initialize the contextual chunk linker.
        
        Args:
            embedder: Text embedder for chunk embeddings
            cache: Optional Redis cache for performance
            spacy_model: spaCy model for NLP processing
            similarity_threshold: Minimum similarity for semantic links
            max_chunk_size: Maximum tokens per chunk
            overlap_size: Token overlap between adjacent chunks
        """
        self.embedder = embedder
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"SpaCy model {spacy_model} not found, using basic tokenization")
            self.nlp = None
        
        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Relationship detection patterns
        self._init_relationship_patterns()
        
        # Graph for link analysis
        self.link_graph = nx.DiGraph()
        
        logger.info(
            "Initialized contextual chunk linker",
            similarity_threshold=similarity_threshold,
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size
        )
    
    def _init_relationship_patterns(self):
        """Initialize patterns for detecting different relationship types."""
        self.causal_patterns = [
            r'\b(?:because|since|due to|as a result|therefore|thus|hence|consequently)\b',
            r'\b(?:causes?|leads? to|results? in|brings? about)\b',
            r'\b(?:if|when|whenever|as soon as).*(?:then|will|would)\b'
        ]
        
        self.temporal_patterns = [
            r'\b(?:before|after|during|while|when|then|next|previously|subsequently)\b',
            r'\b(?:first|second|third|finally|lastly|initially|eventually)\b',
            r'\b(?:earlier|later|meanwhile|simultaneously|concurrent)\b'
        ]
        
        self.reference_patterns = [
            r'\b(?:as mentioned|as stated|as discussed|as shown|as noted)\b',
            r'\b(?:see|refer to|according to|based on|following)\b',
            r'\b(?:above|below|previously|earlier|later)\b'
        ]
        
        self.elaborative_patterns = [
            r'\b(?:for example|for instance|specifically|in particular|namely)\b',
            r'\b(?:furthermore|moreover|additionally|also|in addition)\b',
            r'\b(?:in other words|that is|i\.e\.|e\.g\.)\b'
        ]
        
        self.contradictory_patterns = [
            r'\b(?:however|but|although|despite|nevertheless|nonetheless)\b',
            r'\b(?:on the other hand|in contrast|conversely|whereas)\b',
            r'\b(?:unlike|different from|opposed to)\b'
        ]
    
    async def analyze_chunks_and_links(
        self,
        memories: List[Memory],
        preserve_structure: bool = True
    ) -> ChunkLinkingResult:
        """
        Analyze memories to extract chunks and their contextual relationships.
        
        Args:
            memories: List of memories to analyze
            preserve_structure: Whether to preserve document structure in chunking
            
        Returns:
            Chunk linking analysis result with chunks, links, and metrics
        """
        start_time = datetime.utcnow()
        
        # Step 1: Extract chunks from all memories
        all_chunks = []
        for memory in memories:
            chunks = await self._extract_chunks_from_memory(memory, preserve_structure)
            all_chunks.extend(chunks)
        
        # Step 2: Generate embeddings for all chunks
        await self._generate_chunk_embeddings(all_chunks)
        
        # Step 3: Extract entities and keywords
        await self._extract_chunk_features(all_chunks)
        
        # Step 4: Detect contextual links
        links = await self._detect_contextual_links(all_chunks)
        
        # Step 5: Build and analyze link graph
        graph_metrics = await self._analyze_link_graph(all_chunks, links)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = ChunkLinkingResult(
            chunks=all_chunks,
            links=links,
            graph_metrics=graph_metrics,
            processing_time=processing_time
        )
        
        logger.info(
            "Completed chunk linking analysis",
            chunk_count=len(all_chunks),
            link_count=len(links),
            processing_time=processing_time
        )
        
        return result
    
    async def _extract_chunks_from_memory(
        self, 
        memory: Memory, 
        preserve_structure: bool
    ) -> List[TextChunk]:
        """Extract contextual chunks from a memory."""
        content = memory.content
        
        if preserve_structure and self.nlp:
            return await self._structure_aware_chunking(content, memory.id)
        else:
            return await self._sliding_window_chunking(content, memory.id)
    
    async def _structure_aware_chunking(
        self, 
        content: str, 
        memory_id: str
    ) -> List[TextChunk]:
        """Extract chunks while preserving document structure."""
        chunks = []
        
        # Process with spaCy for sentence and structure detection
        doc = self.nlp(content)
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            # Check if adding this sentence would exceed max chunk size
            if len(current_chunk.split()) + len(sentence_text.split()) > self.max_chunk_size:
                if current_chunk:
                    # Create chunk for accumulated text
                    chunk = TextChunk(
                        id=f"{memory_id}_chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        chunk_type=ChunkType.PARAGRAPH,
                        start_position=current_start,
                        end_position=current_start + len(current_chunk),
                        source_memory_id=memory_id
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk
                current_start = sent.start_char
                current_chunk = sentence_text
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence_text
                else:
                    current_chunk = sentence_text
                    current_start = sent.start_char
        
        # Add final chunk
        if current_chunk:
            chunk = TextChunk(
                id=f"{memory_id}_chunk_{chunk_id}",
                content=current_chunk.strip(),
                chunk_type=ChunkType.PARAGRAPH,
                start_position=current_start,
                end_position=current_start + len(current_chunk),
                source_memory_id=memory_id
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _sliding_window_chunking(
        self, 
        content: str, 
        memory_id: str
    ) -> List[TextChunk]:
        """Extract chunks using sliding window approach."""
        chunks = []
        words = content.split()
        
        if len(words) <= self.max_chunk_size:
            # Single chunk for short content
            chunk = TextChunk(
                id=f"{memory_id}_chunk_0",
                content=content,
                chunk_type=ChunkType.PARAGRAPH,
                start_position=0,
                end_position=len(content),
                source_memory_id=memory_id
            )
            chunks.append(chunk)
            return chunks
        
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.max_chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Calculate character positions
            start_char = len(" ".join(words[:start_idx]))
            if start_idx > 0:
                start_char += 1  # Add space
            end_char = start_char + len(chunk_text)
            
            chunk = TextChunk(
                id=f"{memory_id}_chunk_{chunk_id}",
                content=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                start_position=start_char,
                end_position=end_char,
                source_memory_id=memory_id
            )
            chunks.append(chunk)
            
            chunk_id += 1
            start_idx = end_idx - self.overlap_size
            
            if start_idx >= len(words):
                break
        
        return chunks
    
    async def _generate_chunk_embeddings(self, chunks: List[TextChunk]):
        """Generate embeddings for all chunks."""
        # Batch process embeddings for efficiency
        batch_size = 32
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            # Generate embeddings
            embeddings = await asyncio.gather(*[
                self.embedder.embed(text) for text in batch_texts
            ])
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
    
    async def _extract_chunk_features(self, chunks: List[TextChunk]):
        """Extract entities and keywords from chunks."""
        if not self.nlp:
            return
        
        # Extract all chunk texts for TF-IDF
        chunk_texts = [chunk.content for chunk in chunks]
        
        # Fit TF-IDF vectorizer
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
        except:
            # Fallback for small datasets
            logger.warning("TF-IDF extraction failed, using simple keyword extraction")
            for chunk in chunks:
                chunk.keywords = chunk.content.lower().split()[:10]
            return
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Extract entities with spaCy
            doc = self.nlp(chunk.content)
            chunk.entities = [ent.text for ent in doc.ents]
            
            # Extract top keywords from TF-IDF
            chunk_tfidf = tfidf_matrix[i].toarray()[0]
            top_indices = np.argsort(chunk_tfidf)[-10:][::-1]
            chunk.keywords = [feature_names[idx] for idx in top_indices if chunk_tfidf[idx] > 0]
    
    async def _detect_contextual_links(self, chunks: List[TextChunk]) -> List[ContextualLink]:
        """Detect various types of contextual links between chunks."""
        links = []
        
        # Build pairwise similarity matrix for semantic links
        embeddings = np.array([chunk.embedding for chunk in chunks])
        similarity_matrix = cosine_similarity(embeddings)
        
        # Detect different types of links
        for i, source_chunk in enumerate(chunks):
            for j, target_chunk in enumerate(chunks):
                if i == j:
                    continue
                
                # Sequential links (same document, adjacent chunks)
                if (source_chunk.source_memory_id == target_chunk.source_memory_id and
                    abs(int(source_chunk.id.split('_')[-1]) - int(target_chunk.id.split('_')[-1])) == 1):
                    
                    link = ContextualLink(
                        id=f"seq_{source_chunk.id}_{target_chunk.id}",
                        source_chunk_id=source_chunk.id,
                        target_chunk_id=target_chunk.id,
                        link_type=LinkType.SEQUENTIAL,
                        strength=LinkStrength.STRONG,
                        confidence=0.9,
                        evidence=["Adjacent chunks in document"]
                    )
                    links.append(link)
                
                # Semantic links based on embedding similarity
                similarity = similarity_matrix[i][j]
                if similarity >= self.similarity_threshold:
                    strength = self._calculate_link_strength(similarity)
                    
                    link = ContextualLink(
                        id=f"sem_{source_chunk.id}_{target_chunk.id}",
                        source_chunk_id=source_chunk.id,
                        target_chunk_id=target_chunk.id,
                        link_type=LinkType.SEMANTIC,
                        strength=strength,
                        confidence=similarity,
                        evidence=[f"Cosine similarity: {similarity:.3f}"]
                    )
                    links.append(link)
                
                # Pattern-based relationship detection
                pattern_links = await self._detect_pattern_based_links(source_chunk, target_chunk)
                links.extend(pattern_links)
                
                # Entity-based links
                entity_links = await self._detect_entity_based_links(source_chunk, target_chunk)
                links.extend(entity_links)
        
        return links
    
    async def _detect_pattern_based_links(
        self, 
        source_chunk: TextChunk, 
        target_chunk: TextChunk
    ) -> List[ContextualLink]:
        """Detect links based on linguistic patterns."""
        links = []
        
        source_text = source_chunk.content.lower()
        target_text = target_chunk.content.lower()
        
        # Check for causal relationships
        for pattern in self.causal_patterns:
            if re.search(pattern, source_text) and re.search(pattern, target_text):
                link = ContextualLink(
                    id=f"causal_{source_chunk.id}_{target_chunk.id}",
                    source_chunk_id=source_chunk.id,
                    target_chunk_id=target_chunk.id,
                    link_type=LinkType.CAUSAL,
                    strength=LinkStrength.MODERATE,
                    confidence=0.7,
                    evidence=[f"Causal pattern detected: {pattern}"]
                )
                links.append(link)
        
        # Check for temporal relationships
        for pattern in self.temporal_patterns:
            if re.search(pattern, source_text):
                link = ContextualLink(
                    id=f"temp_{source_chunk.id}_{target_chunk.id}",
                    source_chunk_id=source_chunk.id,
                    target_chunk_id=target_chunk.id,
                    link_type=LinkType.TEMPORAL,
                    strength=LinkStrength.MODERATE,
                    confidence=0.6,
                    evidence=[f"Temporal pattern detected: {pattern}"]
                )
                links.append(link)
        
        # Check for reference relationships
        for pattern in self.reference_patterns:
            if re.search(pattern, source_text):
                link = ContextualLink(
                    id=f"ref_{source_chunk.id}_{target_chunk.id}",
                    source_chunk_id=source_chunk.id,
                    target_chunk_id=target_chunk.id,
                    link_type=LinkType.REFERENCE,
                    strength=LinkStrength.WEAK,
                    confidence=0.5,
                    evidence=[f"Reference pattern detected: {pattern}"]
                )
                links.append(link)
        
        # Check for elaborative relationships
        for pattern in self.elaborative_patterns:
            if re.search(pattern, source_text):
                link = ContextualLink(
                    id=f"elab_{source_chunk.id}_{target_chunk.id}",
                    source_chunk_id=source_chunk.id,
                    target_chunk_id=target_chunk.id,
                    link_type=LinkType.ELABORATIVE,
                    strength=LinkStrength.WEAK,
                    confidence=0.5,
                    evidence=[f"Elaborative pattern detected: {pattern}"]
                )
                links.append(link)
        
        # Check for contradictory relationships
        for pattern in self.contradictory_patterns:
            if re.search(pattern, source_text):
                link = ContextualLink(
                    id=f"contra_{source_chunk.id}_{target_chunk.id}",
                    source_chunk_id=source_chunk.id,
                    target_chunk_id=target_chunk.id,
                    link_type=LinkType.CONTRADICTORY,
                    strength=LinkStrength.MODERATE,
                    confidence=0.6,
                    evidence=[f"Contradictory pattern detected: {pattern}"]
                )
                links.append(link)
        
        return links
    
    async def _detect_entity_based_links(
        self, 
        source_chunk: TextChunk, 
        target_chunk: TextChunk
    ) -> List[ContextualLink]:
        """Detect links based on shared entities."""
        links = []
        
        # Find shared entities
        source_entities = set(ent.lower() for ent in source_chunk.entities)
        target_entities = set(ent.lower() for ent in target_chunk.entities)
        shared_entities = source_entities.intersection(target_entities)
        
        if shared_entities:
            confidence = len(shared_entities) / max(len(source_entities), len(target_entities), 1)
            strength = self._calculate_link_strength(confidence)
            
            link = ContextualLink(
                id=f"ent_{source_chunk.id}_{target_chunk.id}",
                source_chunk_id=source_chunk.id,
                target_chunk_id=target_chunk.id,
                link_type=LinkType.SEMANTIC,
                strength=strength,
                confidence=confidence,
                evidence=[f"Shared entities: {', '.join(shared_entities)}"]
            )
            links.append(link)
        
        return links
    
    def _calculate_link_strength(self, score: float) -> LinkStrength:
        """Calculate link strength from similarity score."""
        if score >= 0.9:
            return LinkStrength.VERY_STRONG
        elif score >= 0.7:
            return LinkStrength.STRONG
        elif score >= 0.5:
            return LinkStrength.MODERATE
        elif score >= 0.3:
            return LinkStrength.WEAK
        else:
            return LinkStrength.VERY_WEAK
    
    async def _analyze_link_graph(
        self, 
        chunks: List[TextChunk], 
        links: List[ContextualLink]
    ) -> Dict[str, float]:
        """Analyze the chunk link graph for insights."""
        # Build NetworkX graph
        self.link_graph.clear()
        
        # Add nodes (chunks)
        for chunk in chunks:
            self.link_graph.add_node(
                chunk.id,
                content=chunk.content[:100],  # Truncated for memory
                chunk_type=chunk.chunk_type.value,
                memory_id=chunk.source_memory_id
            )
        
        # Add edges (links)
        for link in links:
            self.link_graph.add_edge(
                link.source_chunk_id,
                link.target_chunk_id,
                link_type=link.link_type.value,
                strength=link.strength.value,
                confidence=link.confidence,
                weight=link.confidence
            )
        
        # Calculate graph metrics
        metrics = {}
        
        try:
            # Basic metrics
            metrics["node_count"] = self.link_graph.number_of_nodes()
            metrics["edge_count"] = self.link_graph.number_of_edges()
            metrics["density"] = nx.density(self.link_graph)
            
            # Connectivity metrics
            if self.link_graph.number_of_nodes() > 0:
                metrics["avg_clustering"] = nx.average_clustering(self.link_graph.to_undirected())
                
                # Components analysis
                components = list(nx.weakly_connected_components(self.link_graph))
                metrics["component_count"] = len(components)
                metrics["largest_component_size"] = len(max(components, key=len)) if components else 0
                
                # Centrality measures (for smaller graphs)
                if self.link_graph.number_of_nodes() <= 1000:
                    centrality = nx.degree_centrality(self.link_graph)
                    metrics["max_centrality"] = max(centrality.values()) if centrality else 0
                    metrics["avg_centrality"] = np.mean(list(centrality.values())) if centrality else 0
        
        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {e}")
            metrics = {
                "node_count": len(chunks),
                "edge_count": len(links),
                "density": 0.0
            }
        
        return metrics
    
    async def find_contextual_path(
        self,
        source_chunk_id: str,
        target_chunk_id: str,
        max_hops: int = 3
    ) -> Optional[List[str]]:
        """Find the shortest contextual path between two chunks."""
        try:
            if source_chunk_id not in self.link_graph or target_chunk_id not in self.link_graph:
                return None
            
            path = nx.shortest_path(
                self.link_graph,
                source_chunk_id,
                target_chunk_id,
                weight='weight'
            )
            
            if len(path) - 1 <= max_hops:
                return path
            else:
                return None
                
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.warning(f"Path finding failed: {e}")
            return None
    
    async def get_contextual_neighbors(
        self,
        chunk_id: str,
        max_neighbors: int = 5,
        link_types: Optional[List[LinkType]] = None
    ) -> List[Tuple[str, ContextualLink]]:
        """Get contextually related chunks for a given chunk."""
        if chunk_id not in self.link_graph:
            return []
        
        neighbors = []
        
        # Get all outgoing edges
        for target_id in self.link_graph.successors(chunk_id):
            edge_data = self.link_graph[chunk_id][target_id]
            
            # Filter by link type if specified
            if link_types and edge_data['link_type'] not in [lt.value for lt in link_types]:
                continue
            
            # Create ContextualLink object
            link = ContextualLink(
                id=f"{chunk_id}_{target_id}",
                source_chunk_id=chunk_id,
                target_chunk_id=target_id,
                link_type=LinkType(edge_data['link_type']),
                strength=LinkStrength(edge_data['strength']),
                confidence=edge_data['confidence']
            )
            
            neighbors.append((target_id, link))
        
        # Sort by confidence and return top neighbors
        neighbors.sort(key=lambda x: x[1].confidence, reverse=True)
        return neighbors[:max_neighbors]
    
    async def expand_context_for_retrieval(
        self,
        chunk_id: str,
        expansion_hops: int = 2,
        min_link_strength: LinkStrength = LinkStrength.WEAK
    ) -> List[str]:
        """Expand context around a chunk for enhanced retrieval."""
        if chunk_id not in self.link_graph:
            return [chunk_id]
        
        # BFS expansion with link strength filtering
        visited = set()
        queue = deque([(chunk_id, 0)])
        expanded_chunks = []
        
        while queue:
            current_id, hops = queue.popleft()
            
            if current_id in visited or hops > expansion_hops:
                continue
            
            visited.add(current_id)
            expanded_chunks.append(current_id)
            
            # Add neighbors if within hop limit
            if hops < expansion_hops:
                for neighbor_id in self.link_graph.successors(current_id):
                    edge_data = self.link_graph[current_id][neighbor_id]
                    link_strength = LinkStrength(edge_data['strength'])
                    
                    # Check if link strength meets minimum requirement
                    strength_values = {
                        LinkStrength.VERY_WEAK: 1,
                        LinkStrength.WEAK: 2,
                        LinkStrength.MODERATE: 3,
                        LinkStrength.STRONG: 4,
                        LinkStrength.VERY_STRONG: 5
                    }
                    
                    if strength_values[link_strength] >= strength_values[min_link_strength]:
                        queue.append((neighbor_id, hops + 1))
        
        return expanded_chunks
    
    async def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about chunks and links."""
        if not self.link_graph:
            return {}
        
        stats = {}
        
        # Link type distribution
        link_types = defaultdict(int)
        link_strengths = defaultdict(int)
        
        for _, _, edge_data in self.link_graph.edges(data=True):
            link_types[edge_data['link_type']] += 1
            link_strengths[edge_data['strength']] += 1
        
        stats['link_type_distribution'] = dict(link_types)
        stats['link_strength_distribution'] = dict(link_strengths)
        
        # Graph connectivity
        stats['graph_density'] = nx.density(self.link_graph)
        stats['is_connected'] = nx.is_weakly_connected(self.link_graph)
        
        return stats