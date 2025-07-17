"""
Memory Deduplication Engine with Semantic Similarity.

This module provides advanced deduplication capabilities using semantic similarity,
hash-based fingerprinting, and intelligent merge strategies. All operations are
performed locally with zero external API calls.
"""

import hashlib
import asyncio
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import structlog
from pydantic import BaseModel, Field, ConfigDict
import json
from collections import defaultdict

from ...models.memory import Memory, MemoryUpdate
from ..embeddings.embedder import Embedder
from ..cache.redis_cache import RedisCache
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class DuplicateType(str, Enum):
    """Types of duplicate memories."""
    EXACT = "exact"  # Identical content
    SEMANTIC = "semantic"  # Similar meaning
    PARTIAL = "partial"  # Overlapping content
    SUPERSEDED = "superseded"  # Newer version of same information


class MergeStrategy(str, Enum):
    """Strategies for merging duplicate memories."""
    KEEP_NEWEST = "keep_newest"
    KEEP_OLDEST = "keep_oldest"
    MERGE_CONTENT = "merge_content"
    CREATE_SUMMARY = "create_summary"
    USER_CHOICE = "user_choice"


@dataclass
class DuplicateGroup:
    """Group of duplicate memories."""
    duplicate_type: DuplicateType
    memories: List[Memory]
    similarity_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    merge_strategy: Optional[MergeStrategy] = None
    merged_memory: Optional[Memory] = None


class DuplicationMetrics(BaseModel):
    """Metrics for deduplication analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    total_memories: int = Field(description="Total number of memories analyzed")
    duplicate_groups: int = Field(description="Number of duplicate groups found")
    exact_duplicates: int = Field(description="Number of exact duplicates")
    semantic_duplicates: int = Field(description="Number of semantic duplicates")
    memories_reduced: int = Field(description="Number of memories after deduplication")
    reduction_percentage: float = Field(description="Percentage reduction in memories")
    processing_time_ms: float = Field(description="Time taken for deduplication in milliseconds")


class DeduplicationEngine:
    """
    Advanced memory deduplication engine using semantic similarity and fingerprinting.
    
    Features:
    - Hash-based exact duplicate detection
    - Semantic similarity using sentence-transformers
    - Multiple merge strategies
    - Caching for performance optimization
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        embedder: Embedder,
        cache: Optional[RedisCache] = None,
        similarity_threshold: float = 0.85,
        batch_size: int = 32
    ):
        """
        Initialize the deduplication engine.
        
        Args:
            embedder: Embedder instance for generating embeddings
            cache: Optional Redis cache for performance
            similarity_threshold: Threshold for semantic similarity (0.85 = 85% similar)
            batch_size: Batch size for processing memories
        """
        self.embedder = embedder
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Initialize local sentence transformer for fallback
        self.sentence_model = None
        self._init_sentence_transformer()
        
        # TF-IDF for partial matching
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        logger.info(
            "Initialized deduplication engine",
            similarity_threshold=similarity_threshold,
            batch_size=batch_size
        )
    
    def _init_sentence_transformer(self):
        """Initialize local sentence transformer model."""
        try:
            model_name = settings.synthesis.deduplication.fallback_model
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for exact matching."""
        return hashlib.sha256(content.strip().lower().encode()).hexdigest()
    
    async def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts with caching."""
        embeddings = []
        cache_keys = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        if self.cache:
            for i, text in enumerate(texts):
                cache_key = f"dedup_embedding:{self._compute_hash(text)}"
                cache_keys.append(cache_key)
                cached = await self.cache.get(cache_key)
                if cached is not None:
                    embeddings.append(np.array(cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                # Try primary embedder first
                new_embeddings = await self.embedder.embed_batch(uncached_texts)
                
                # Cache the results
                if self.cache:
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        await self.cache.set(
                            cache_keys[idx],
                            embedding.tolist(),
                            ttl=86400  # 24 hours
                        )
                
                # Merge cached and new embeddings
                if embeddings:
                    final_embeddings = np.zeros((len(texts), new_embeddings.shape[1]))
                    cached_idx = 0
                    new_idx = 0
                    for i in range(len(texts)):
                        if i in uncached_indices:
                            final_embeddings[i] = new_embeddings[new_idx]
                            new_idx += 1
                        else:
                            final_embeddings[i] = embeddings[cached_idx]
                            cached_idx += 1
                    return final_embeddings
                else:
                    return new_embeddings
                    
            except Exception as e:
                logger.warning(f"Primary embedder failed, using fallback: {e}")
                
                # Fallback to sentence transformer
                if self.sentence_model:
                    embeddings = self.sentence_model.encode(
                        uncached_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    return embeddings.cpu().numpy()
                else:
                    raise
        
        return np.array(embeddings)
    
    async def find_duplicates(
        self,
        memories: List[Memory],
        check_types: Optional[List[DuplicateType]] = None
    ) -> List[DuplicateGroup]:
        """
        Find duplicate memories using multiple strategies.
        
        Args:
            memories: List of memories to analyze
            check_types: Types of duplicates to check for (default: all types)
            
        Returns:
            List of duplicate groups found
        """
        if not memories:
            return []
        
        start_time = datetime.utcnow()
        
        if check_types is None:
            check_types = list(DuplicateType)
        
        duplicate_groups = []
        processed_ids = set()
        
        # Step 1: Find exact duplicates using hashing
        if DuplicateType.EXACT in check_types:
            exact_groups = await self._find_exact_duplicates(memories)
            duplicate_groups.extend(exact_groups)
            for group in exact_groups:
                processed_ids.update(m.id for m in group.memories)
        
        # Step 2: Find semantic duplicates using embeddings
        if DuplicateType.SEMANTIC in check_types:
            unprocessed = [m for m in memories if m.id not in processed_ids]
            if unprocessed:
                semantic_groups = await self._find_semantic_duplicates(unprocessed)
                duplicate_groups.extend(semantic_groups)
                for group in semantic_groups:
                    processed_ids.update(m.id for m in group.memories)
        
        # Step 3: Find partial duplicates using TF-IDF
        if DuplicateType.PARTIAL in check_types:
            unprocessed = [m for m in memories if m.id not in processed_ids]
            if unprocessed:
                partial_groups = await self._find_partial_duplicates(unprocessed)
                duplicate_groups.extend(partial_groups)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info(
            "Deduplication analysis complete",
            total_memories=len(memories),
            duplicate_groups=len(duplicate_groups),
            processing_time_ms=processing_time
        )
        
        return duplicate_groups
    
    async def _find_exact_duplicates(self, memories: List[Memory]) -> List[DuplicateGroup]:
        """Find exact duplicates using content hashing."""
        hash_groups = defaultdict(list)
        
        for memory in memories:
            content_hash = self._compute_hash(memory.content)
            hash_groups[content_hash].append(memory)
        
        duplicate_groups = []
        for hash_value, group_memories in hash_groups.items():
            if len(group_memories) > 1:
                # Sort by timestamp to identify newest/oldest
                group_memories.sort(key=lambda m: m.created_at)
                
                group = DuplicateGroup(
                    duplicate_type=DuplicateType.EXACT,
                    memories=group_memories,
                    merge_strategy=MergeStrategy.KEEP_NEWEST
                )
                
                # Add similarity scores (1.0 for exact matches)
                for i in range(len(group_memories)):
                    for j in range(i + 1, len(group_memories)):
                        key = (group_memories[i].id, group_memories[j].id)
                        group.similarity_scores[key] = 1.0
                
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    async def _find_semantic_duplicates(self, memories: List[Memory]) -> List[DuplicateGroup]:
        """Find semantically similar memories using embeddings."""
        if not memories:
            return []
        
        duplicate_groups = []
        processed = set()
        
        # Process in batches for efficiency
        for batch_start in range(0, len(memories), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(memories))
            batch_memories = memories[batch_start:batch_end]
            
            # Skip already processed memories
            batch_memories = [m for m in batch_memories if m.id not in processed]
            if not batch_memories:
                continue
            
            # Get embeddings for batch
            texts = [m.content for m in batch_memories]
            embeddings = await self._get_embeddings_batch(texts)
            
            # Compute similarity matrix
            if isinstance(embeddings, torch.Tensor):
                similarities = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
            else:
                similarities = cosine_similarity(embeddings)
            
            # Find similar pairs
            for i in range(len(batch_memories)):
                if batch_memories[i].id in processed:
                    continue
                
                similar_indices = []
                similarity_scores = {}
                
                for j in range(i + 1, len(batch_memories)):
                    if similarities[i][j] >= self.similarity_threshold:
                        similar_indices.append(j)
                        key = (batch_memories[i].id, batch_memories[j].id)
                        similarity_scores[key] = float(similarities[i][j])
                
                if similar_indices:
                    # Create duplicate group
                    group_memories = [batch_memories[i]]
                    group_memories.extend([batch_memories[j] for j in similar_indices])
                    
                    group = DuplicateGroup(
                        duplicate_type=DuplicateType.SEMANTIC,
                        memories=group_memories,
                        similarity_scores=similarity_scores,
                        merge_strategy=MergeStrategy.MERGE_CONTENT
                    )
                    
                    duplicate_groups.append(group)
                    processed.update(m.id for m in group_memories)
        
        return duplicate_groups
    
    async def _find_partial_duplicates(self, memories: List[Memory]) -> List[DuplicateGroup]:
        """Find partial duplicates using TF-IDF and n-gram matching."""
        if len(memories) < 2:
            return []
        
        texts = [m.content for m in memories]
        
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Compute cosine similarity
            similarities = cosine_similarity(tfidf_matrix)
            
            duplicate_groups = []
            processed = set()
            
            # Use lower threshold for partial matches
            partial_threshold = self.similarity_threshold * 0.7
            
            for i in range(len(memories)):
                if memories[i].id in processed:
                    continue
                
                similar_indices = []
                similarity_scores = {}
                
                for j in range(i + 1, len(memories)):
                    if similarities[i][j] >= partial_threshold:
                        similar_indices.append(j)
                        key = (memories[i].id, memories[j].id)
                        similarity_scores[key] = float(similarities[i][j])
                
                if similar_indices:
                    group_memories = [memories[i]]
                    group_memories.extend([memories[j] for j in similar_indices])
                    
                    group = DuplicateGroup(
                        duplicate_type=DuplicateType.PARTIAL,
                        memories=group_memories,
                        similarity_scores=similarity_scores,
                        merge_strategy=MergeStrategy.USER_CHOICE
                    )
                    
                    duplicate_groups.append(group)
                    processed.update(m.id for m in group_memories)
            
            return duplicate_groups
            
        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {e}")
            return []
    
    async def merge_duplicates(
        self,
        duplicate_group: DuplicateGroup,
        strategy: Optional[MergeStrategy] = None,
        user_choice: Optional[Memory] = None
    ) -> Memory:
        """
        Merge duplicate memories using the specified strategy.
        
        Args:
            duplicate_group: Group of duplicate memories
            strategy: Merge strategy to use (overrides group strategy)
            user_choice: User's choice for USER_CHOICE strategy
            
        Returns:
            Merged memory
        """
        strategy = strategy or duplicate_group.merge_strategy or MergeStrategy.KEEP_NEWEST
        memories = duplicate_group.memories
        
        if not memories:
            raise ValueError("No memories to merge")
        
        # Sort by creation date
        memories.sort(key=lambda m: m.created_at)
        
        if strategy == MergeStrategy.KEEP_NEWEST:
            merged = memories[-1]
            
        elif strategy == MergeStrategy.KEEP_OLDEST:
            merged = memories[0]
            
        elif strategy == MergeStrategy.USER_CHOICE:
            if not user_choice or user_choice not in memories:
                raise ValueError("Valid user choice required for USER_CHOICE strategy")
            merged = user_choice
            
        elif strategy == MergeStrategy.MERGE_CONTENT:
            # Intelligently merge content
            merged_content = await self._merge_content(memories)
            
            # Use metadata from newest memory
            base_memory = memories[-1]
            merged = Memory(
                id=base_memory.id,
                user_id=base_memory.user_id,
                content=merged_content,
                metadata={
                    **base_memory.metadata,
                    "merged_from": [m.id for m in memories if m.id != base_memory.id],
                    "merge_strategy": strategy.value,
                    "original_count": len(memories)
                },
                created_at=base_memory.created_at,
                updated_at=datetime.utcnow()
            )
            
        elif strategy == MergeStrategy.CREATE_SUMMARY:
            # Create a summary of all memories
            summary = await self._create_summary(memories)
            
            base_memory = memories[-1]
            merged = Memory(
                id=base_memory.id,
                user_id=base_memory.user_id,
                content=summary,
                metadata={
                    **base_memory.metadata,
                    "summarized_from": [m.id for m in memories],
                    "merge_strategy": strategy.value,
                    "original_count": len(memories)
                },
                created_at=base_memory.created_at,
                updated_at=datetime.utcnow()
            )
        
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        duplicate_group.merged_memory = merged
        return merged
    
    async def _merge_content(self, memories: List[Memory]) -> str:
        """Intelligently merge content from multiple memories."""
        # Extract unique information from each memory
        contents = [m.content for m in memories]
        
        # Simple strategy: combine unique sentences
        all_sentences = []
        seen_sentences = set()
        
        for content in contents:
            sentences = content.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence.lower() not in seen_sentences:
                    all_sentences.append(sentence)
                    seen_sentences.add(sentence.lower())
        
        # Join sentences back together
        merged_content = '. '.join(all_sentences)
        if not merged_content.endswith('.'):
            merged_content += '.'
        
        return merged_content
    
    async def _create_summary(self, memories: List[Memory]) -> str:
        """Create a summary of multiple memories."""
        # For now, use a simple extractive summary
        # In production, this would use a local summarization model
        
        contents = [m.content for m in memories]
        combined = ' '.join(contents)
        
        # Extract key sentences (simple extractive approach)
        sentences = combined.split('. ')
        
        # Limit to first few sentences as summary
        summary_sentences = sentences[:min(3, len(sentences))]
        summary = '. '.join(summary_sentences)
        
        if not summary.endswith('.'):
            summary += '.'
        
        return f"Summary of {len(memories)} related memories: {summary}"
    
    async def get_deduplication_metrics(
        self,
        memories: List[Memory],
        duplicate_groups: List[DuplicateGroup]
    ) -> DuplicationMetrics:
        """Calculate comprehensive deduplication metrics."""
        total_memories = len(memories)
        
        # Count duplicates by type
        exact_duplicates = sum(
            len(g.memories) - 1 
            for g in duplicate_groups 
            if g.duplicate_type == DuplicateType.EXACT
        )
        
        semantic_duplicates = sum(
            len(g.memories) - 1 
            for g in duplicate_groups 
            if g.duplicate_type == DuplicateType.SEMANTIC
        )
        
        # Calculate reduction
        total_duplicates = sum(len(g.memories) - 1 for g in duplicate_groups)
        memories_reduced = total_memories - total_duplicates
        reduction_percentage = (total_duplicates / total_memories * 100) if total_memories > 0 else 0
        
        return DuplicationMetrics(
            total_memories=total_memories,
            duplicate_groups=len(duplicate_groups),
            exact_duplicates=exact_duplicates,
            semantic_duplicates=semantic_duplicates,
            memories_reduced=memories_reduced,
            reduction_percentage=reduction_percentage,
            processing_time_ms=0  # Set by caller
        )
    
    async def suggest_merge_strategy(
        self,
        duplicate_group: DuplicateGroup
    ) -> MergeStrategy:
        """Suggest the best merge strategy for a duplicate group."""
        memories = duplicate_group.memories
        
        # For exact duplicates, keep newest
        if duplicate_group.duplicate_type == DuplicateType.EXACT:
            return MergeStrategy.KEEP_NEWEST
        
        # For semantic duplicates with high similarity, merge content
        avg_similarity = np.mean(list(duplicate_group.similarity_scores.values()))
        if avg_similarity >= 0.9:
            return MergeStrategy.MERGE_CONTENT
        
        # For partial duplicates or lower similarity, let user choose
        if duplicate_group.duplicate_type == DuplicateType.PARTIAL:
            return MergeStrategy.USER_CHOICE
        
        # For groups with many memories, create summary
        if len(memories) > 5:
            return MergeStrategy.CREATE_SUMMARY
        
        # Default to merging content
        return MergeStrategy.MERGE_CONTENT
    
    async def get_user_confirmation_for_merge(
        self,
        duplicate_group: DuplicateGroup,
        suggested_strategy: MergeStrategy,
        interactive: bool = True
    ) -> Tuple[bool, MergeStrategy]:
        """
        Get user confirmation for suggested merge with CLI interface.
        
        Args:
            duplicate_group: Group of duplicate memories
            suggested_strategy: AI-suggested merge strategy
            interactive: Whether to use interactive CLI (False for automated processing)
            
        Returns:
            Tuple of (user_approved, final_strategy)
        """
        if not interactive:
            # Auto-approve high-confidence exact duplicates
            if (duplicate_group.duplicate_type == DuplicateType.EXACT and 
                suggested_strategy == MergeStrategy.KEEP_NEWEST):
                return True, suggested_strategy
            
            # For non-interactive mode, be conservative
            avg_similarity = np.mean(list(duplicate_group.similarity_scores.values()))
            if avg_similarity >= 0.95:
                return True, MergeStrategy.MERGE_CONTENT
            else:
                return False, suggested_strategy
        
        try:
            # Display duplicate information
            print("\n" + "="*80)
            print("🔍 DUPLICATE MEMORIES DETECTED")
            print("="*80)
            
            print(f"\nDuplicate Type: {duplicate_group.duplicate_type.value.upper()}")
            print(f"Suggested Strategy: {suggested_strategy.value.replace('_', ' ').title()}")
            print(f"Number of memories: {len(duplicate_group.memories)}")
            
            # Show similarity scores
            if duplicate_group.similarity_scores:
                avg_similarity = np.mean(list(duplicate_group.similarity_scores.values()))
                print(f"Average similarity: {avg_similarity:.3f}")
            
            print("\nMemories found:")
            print("-" * 40)
            
            for i, memory in enumerate(duplicate_group.memories, 1):
                content_preview = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content
                created_at = memory.created_at.strftime("%Y-%m-%d %H:%M:%S") if memory.created_at else "Unknown"
                
                print(f"\n{i}. Memory ID: {memory.id}")
                print(f"   Created: {created_at}")
                print(f"   Content: {content_preview}")
                
                # Show tags if available
                if hasattr(memory, 'tags') and memory.tags:
                    print(f"   Tags: {', '.join(memory.tags)}")
            
            print("\n" + "-" * 80)
            print("MERGE OPTIONS:")
            print("1. Accept suggested strategy")
            print("2. Keep newest memory only")
            print("3. Keep oldest memory only") 
            print("4. Merge all content together")
            print("5. Create summary of all memories")
            print("6. Skip this group (no merge)")
            print("7. Show detailed comparison")
            
            while True:
                try:
                    choice = input("\nEnter your choice (1-7): ").strip()
                    
                    if choice == "1":
                        print(f"✅ Accepted suggested strategy: {suggested_strategy.value}")
                        return True, suggested_strategy
                    
                    elif choice == "2":
                        print("✅ Will keep newest memory only")
                        return True, MergeStrategy.KEEP_NEWEST
                    
                    elif choice == "3":
                        print("✅ Will keep oldest memory only")
                        return True, MergeStrategy.KEEP_OLDEST
                    
                    elif choice == "4":
                        print("✅ Will merge all content together")
                        return True, MergeStrategy.MERGE_CONTENT
                    
                    elif choice == "5":
                        print("✅ Will create summary of all memories")
                        return True, MergeStrategy.CREATE_SUMMARY
                    
                    elif choice == "6":
                        print("⏭️ Skipping this group")
                        return False, suggested_strategy
                    
                    elif choice == "7":
                        await self._show_detailed_comparison(duplicate_group)
                        continue
                    
                    else:
                        print("❌ Invalid choice. Please enter a number from 1-7.")
                        continue
                        
                except KeyboardInterrupt:
                    print("\n\n⏹️ Merge confirmation cancelled by user")
                    return False, suggested_strategy
                except EOFError:
                    print("\n⏹️ Input stream ended")
                    return False, suggested_strategy
                    
        except Exception as e:
            logger.error("Error in user confirmation", error=str(e))
            print(f"\n❌ Error during confirmation: {e}")
            print("Defaulting to no merge for safety")
            return False, suggested_strategy
    
    async def _show_detailed_comparison(self, duplicate_group: DuplicateGroup) -> None:
        """Show detailed side-by-side comparison of duplicate memories."""
        print("\n" + "="*100)
        print("DETAILED COMPARISON")
        print("="*100)
        
        memories = duplicate_group.memories
        
        # Show memories side by side
        for i in range(0, len(memories), 2):
            left_memory = memories[i]
            right_memory = memories[i + 1] if i + 1 < len(memories) else None
            
            print(f"\n{'Memory ' + str(i+1):<50} {'Memory ' + str(i+2) if right_memory else ''}")
            print("-" * 100)
            
            # Memory IDs
            left_id = f"ID: {left_memory.id}"
            right_id = f"ID: {right_memory.id}" if right_memory else ""
            print(f"{left_id:<50} {right_id}")
            
            # Creation dates
            left_date = left_memory.created_at.strftime("%Y-%m-%d %H:%M:%S") if left_memory.created_at else "Unknown"
            right_date = right_memory.created_at.strftime("%Y-%m-%d %H:%M:%S") if right_memory and right_memory.created_at else "Unknown"
            print(f"Created: {left_date:<42} Created: {right_date}")
            
            # Content comparison
            left_lines = left_memory.content.split('\n')
            right_lines = right_memory.content.split('\n') if right_memory else []
            
            max_lines = max(len(left_lines), len(right_lines))
            
            print("\nContent:")
            for line_idx in range(min(max_lines, 10)):  # Show first 10 lines
                left_line = left_lines[line_idx] if line_idx < len(left_lines) else ""
                right_line = right_lines[line_idx] if line_idx < len(right_lines) else ""
                
                # Truncate long lines
                left_line = left_line[:45] + "..." if len(left_line) > 45 else left_line
                right_line = right_line[:45] + "..." if len(right_line) > 45 else right_line
                
                print(f"{left_line:<50} {right_line}")
            
            if max_lines > 10:
                print("... (content truncated)")
            
            # Show similarity if available
            if right_memory and duplicate_group.similarity_scores:
                pair_key = f"{left_memory.id}_{right_memory.id}"
                similarity = duplicate_group.similarity_scores.get(pair_key, 0.0)
                print(f"\nSimilarity score: {similarity:.3f}")
            
            print("\n" + "="*100)
        
        input("\nPress Enter to continue...")
    
    async def batch_process_duplicates_with_confirmation(
        self,
        memories: List[Memory],
        interactive: bool = True,
        auto_approve_threshold: float = 0.95
    ) -> Tuple[List[DuplicateGroup], List[DuplicateGroup]]:
        """
        Process all duplicates with user confirmation for each group.
        
        Args:
            memories: List of memories to check for duplicates
            interactive: Whether to show CLI prompts
            auto_approve_threshold: Similarity threshold for auto-approval
            
        Returns:
            Tuple of (approved_groups, skipped_groups)
        """
        try:
            # Find all duplicate groups
            duplicate_groups = await self.find_duplicates(memories)
            
            if not duplicate_groups:
                print("\n✅ No duplicate memories found!")
                return [], []
            
            print(f"\n🔍 Found {len(duplicate_groups)} groups of duplicate memories")
            
            approved_groups = []
            skipped_groups = []
            
            for i, group in enumerate(duplicate_groups, 1):
                print(f"\n📋 Processing group {i} of {len(duplicate_groups)}")
                
                # Get suggested strategy
                suggested_strategy = await self.suggest_merge_strategy(group)
                
                # Auto-approve high-confidence cases if threshold met
                if not interactive:
                    avg_similarity = np.mean(list(group.similarity_scores.values())) if group.similarity_scores else 0.0
                    if avg_similarity >= auto_approve_threshold:
                        approved_groups.append(group)
                        continue
                    else:
                        skipped_groups.append(group)
                        continue
                
                # Get user confirmation
                approved, final_strategy = await self.get_user_confirmation_for_merge(
                    group, suggested_strategy, interactive
                )
                
                if approved:
                    # Update group with final strategy
                    group.suggested_strategy = final_strategy
                    approved_groups.append(group)
                    print(f"✅ Group {i} approved for merge")
                else:
                    skipped_groups.append(group)
                    print(f"⏭️ Group {i} skipped")
            
            print(f"\n📊 SUMMARY:")
            print(f"   Groups approved: {len(approved_groups)}")
            print(f"   Groups skipped: {len(skipped_groups)}")
            print(f"   Total memories that will be processed: {sum(len(g.memories) for g in approved_groups)}")
            
            if approved_groups and interactive:
                final_confirm = input("\nProceed with approved merges? (y/N): ").strip().lower()
                if final_confirm not in ['y', 'yes']:
                    print("🚫 Merge operation cancelled by user")
                    return [], duplicate_groups
            
            return approved_groups, skipped_groups
            
        except Exception as e:
            logger.error("Error in batch duplicate processing", error=str(e))
            print(f"\n❌ Error during batch processing: {e}")
            return [], duplicate_groups if 'duplicate_groups' in locals() else []