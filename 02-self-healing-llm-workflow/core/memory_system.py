"""
MemorySystem: Advanced memory management for workflow learning

This system implements multi-layered memory storage for workflow optimization,
including short-term execution memory, long-term pattern memory, and 
semantic knowledge extraction.
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import os

from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Expert-level configuration
class ExpertConfig:
    def get(self, key, default=None):
        config_map = {
            'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'memory_capacity': 10000,
            'similarity_threshold': 0.7
        }
        return config_map.get(key, default)

class ExpertMetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.counters = {}
        self.request_times = {}
        
    def track_request(self, service, operation):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.record_metric(f"{service}_{operation}_success", 1, {"service": service, "operation": operation})
                    self.record_metric(f"{service}_{operation}_duration", execution_time, {"service": service, "operation": operation})
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.record_metric(f"{service}_{operation}_error", 1, {"service": service, "operation": operation})
                    self.record_metric(f"{service}_{operation}_duration", execution_time, {"service": service, "operation": operation})
                    raise
            return wrapper
        return decorator
    
    def record_metric(self, name, value, tags=None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            'value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        })
        
    def increment_counter(self, name, value=1, tags=None):
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        
    def get_metrics(self):
        return {
            'metrics': self.metrics,
            'counters': self.counters,
            'request_times': self.request_times
        }

config_instance = ExpertConfig()
metrics_collector = ExpertMetricsCollector()

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    PATTERN = "pattern"

@dataclass
class MemoryRecord:
    """Represents a memory record in the system"""
    id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    access_count: int = 0
    importance_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowPattern:
    """Represents a learned workflow pattern"""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    success_rate: float
    usage_count: int = 0
    confidence: float = 0.0
    examples: List[str] = field(default_factory=list)

class MemorySystem:
    """
    Advanced memory system for workflow learning and optimization
    
    Features:
    - Multi-layered memory architecture
    - Semantic similarity search
    - Pattern recognition and extraction
    - Automatic memory consolidation
    - Importance-based retention
    - Real-time learning updates
    """
    
    def __init__(self):
        self.config = config_instance
        self.logger = logging.getLogger(__name__)
        self.metrics = metrics_collector
        
        # Memory storage
        self.memories: Dict[str, MemoryRecord] = {}
        self.memory_indices: Dict[MemoryType, List[str]] = defaultdict(list)
        
        # Pattern storage
        self.patterns: Dict[str, WorkflowPattern] = {}
        self.pattern_index = defaultdict(list)
        
        # Semantic search setup
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        self.semantic_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.semantic_id_map: List[str] = []
        
        # Working memory (temporary, high-speed access)
        self.working_memory = deque(maxlen=1000)
        
        # Configuration
        self.max_short_term_memories = self.config.get('memory', {}).get('max_short_term', 10000)
        self.max_long_term_memories = self.config.get('memory', {}).get('max_long_term', 50000)
        self.consolidation_threshold = self.config.get('memory', {}).get('consolidation_threshold', 100)
        self.importance_decay_rate = self.config.get('memory', {}).get('importance_decay', 0.95)
        
        # Performance tracking
        self.total_memories = 0
        self.memory_hits = 0
        self.memory_misses = 0
        
        self.logger.info("MemorySystem initialized successfully")

    async def store_execution_knowledge(self, execution) -> List[str]:
        """
        Store knowledge extracted from a workflow execution
        
        Args:
            execution: WorkflowExecution object
            
        Returns:
            List of memory record IDs created
        """
        try:
            created_records = []
            
            # Extract and store different types of knowledge
            
            # 1. Store execution summary
            execution_summary = self._extract_execution_summary(execution)
            summary_id = await self._store_memory(
                MemoryType.SHORT_TERM,
                execution_summary,
                importance=0.8
            )
            created_records.append(summary_id)
            
            # 2. Store task-level knowledge
            for task in execution.tasks:
                task_knowledge = self._extract_task_knowledge(task)
                task_id = await self._store_memory(
                    MemoryType.WORKING,
                    task_knowledge,
                    importance=0.6
                )
                created_records.append(task_id)
            
            # 3. Store error patterns if any
            if execution.healing_events:
                error_patterns = self._extract_error_patterns(execution)
                for pattern in error_patterns:
                    pattern_id = await self._store_memory(
                        MemoryType.PATTERN,
                        pattern,
                        importance=0.9
                    )
                    created_records.append(pattern_id)
            
            # 4. Extract and store workflow patterns
            workflow_patterns = await self._extract_workflow_patterns(execution)
            for pattern in workflow_patterns:
                await self._store_pattern(pattern)
            
            # 5. Consolidate memories if threshold reached
            if len(self.memories) > self.consolidation_threshold:
                await self._consolidate_memories()
            
            self.logger.info(f"Stored {len(created_records)} memory records from execution {execution.id}")
            self.metrics.increment_counter('memories_stored', len(created_records))
            
            return created_records
            
        except Exception as e:
            self.logger.error(f"Failed to store execution knowledge: {str(e)}")
            raise

    async def retrieve_relevant_knowledge(self, 
                                        query: str, 
                                        memory_types: Optional[List[MemoryType]] = None,
                                        limit: int = 10) -> List[MemoryRecord]:
        """
        Retrieve relevant knowledge based on semantic similarity
        
        Args:
            query: Search query
            memory_types: Types of memory to search (None for all)
            limit: Maximum number of results
            
        Returns:
            List of relevant memory records
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search semantic index
            scores, indices = self.semantic_index.search(query_embedding, min(limit * 2, len(self.semantic_id_map)))
            
            # Filter and rank results
            relevant_memories = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.semantic_id_map):
                    memory_id = self.semantic_id_map[idx]
                    if memory_id in self.memories:
                        memory = self.memories[memory_id]
                        
                        # Filter by memory type if specified
                        if memory_types and memory.memory_type not in memory_types:
                            continue
                        
                        # Update access statistics
                        memory.access_count += 1
                        memory.last_accessed = time.time()
                        
                        # Calculate relevance score
                        relevance_score = float(score) * memory.importance_score
                        memory.metadata['relevance_score'] = relevance_score
                        
                        relevant_memories.append(memory)
                        
                        if len(relevant_memories) >= limit:
                            break
            
            # Update hit/miss statistics
            if relevant_memories:
                self.memory_hits += 1
            else:
                self.memory_misses += 1
            
            self.logger.debug(f"Retrieved {len(relevant_memories)} relevant memories for query: {query[:50]}...")
            return relevant_memories
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve relevant knowledge: {str(e)}")
            return []

    async def get_workflow_patterns(self, 
                                  conditions: Dict[str, Any], 
                                  pattern_type: Optional[str] = None) -> List[WorkflowPattern]:
        """
        Get workflow patterns matching specified conditions
        
        Args:
            conditions: Conditions to match against
            pattern_type: Type of pattern to search for
            
        Returns:
            List of matching workflow patterns
        """
        try:
            matching_patterns = []
            
            for pattern in self.patterns.values():
                # Filter by pattern type if specified
                if pattern_type and pattern.pattern_type != pattern_type:
                    continue
                
                # Check if pattern conditions match
                if self._pattern_matches_conditions(pattern, conditions):
                    pattern.usage_count += 1
                    matching_patterns.append(pattern)
            
            # Sort by success rate and confidence
            matching_patterns.sort(key=lambda p: (p.success_rate, p.confidence), reverse=True)
            
            return matching_patterns
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow patterns: {str(e)}")
            return []

    async def update_pattern_performance(self, pattern_id: str, success: bool, execution_time: float):
        """Update pattern performance based on usage results"""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            # Update success rate using weighted average
            total_uses = pattern.usage_count
            if total_uses > 0:
                current_success_rate = pattern.success_rate
                if success:
                    new_success_rate = (current_success_rate * (total_uses - 1) + 1.0) / total_uses
                else:
                    new_success_rate = (current_success_rate * (total_uses - 1)) / total_uses
                pattern.success_rate = new_success_rate
            
            # Update confidence based on usage count
            pattern.confidence = min(1.0, pattern.usage_count / 10.0)

    async def _store_memory(self, 
                          memory_type: MemoryType, 
                          content: Dict[str, Any], 
                          importance: float = 0.5) -> str:
        """Store a memory record"""
        # Generate unique ID
        content_str = json.dumps(content, sort_keys=True)
        memory_id = hashlib.md5(content_str.encode()).hexdigest()
        
        # Generate embeddings for semantic search
        text_content = self._extract_text_for_embedding(content)
        embeddings = self.embedding_model.encode([text_content])
        
        # Create memory record
        memory = MemoryRecord(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            embeddings=embeddings[0],
            importance_score=importance
        )
        
        # Store memory
        self.memories[memory_id] = memory
        self.memory_indices[memory_type].append(memory_id)
        
        # Add to semantic index
        faiss.normalize_L2(embeddings)
        self.semantic_index.add(embeddings)
        self.semantic_id_map.append(memory_id)
        
        self.total_memories += 1
        
        return memory_id

    async def _store_pattern(self, pattern: WorkflowPattern):
        """Store a workflow pattern"""
        self.patterns[pattern.pattern_id] = pattern
        self.pattern_index[pattern.pattern_type].append(pattern.pattern_id)

    def _extract_execution_summary(self, execution) -> Dict[str, Any]:
        """Extract high-level summary from execution"""
        return {
            'execution_id': execution.id,
            'workflow_name': execution.name,
            'status': execution.status.value,
            'total_tasks': len(execution.tasks),
            'success_rate': execution.success_rate,
            'execution_time': execution.total_execution_time,
            'healing_events': len(execution.healing_events),
            'task_types': list(set(task.agent_type for task in execution.tasks)),
            'metadata': execution.metadata
        }

    def _extract_task_knowledge(self, task) -> Dict[str, Any]:
        """Extract knowledge from individual task"""
        return {
            'task_id': task.id,
            'task_name': task.name,
            'agent_type': task.agent_type,
            'status': task.status.value,
            'execution_time': task.execution_time,
            'retry_count': task.retry_count,
            'input_summary': self._summarize_data(task.input_data),
            'output_summary': self._summarize_data(task.output_data),
            'error_info': task.error_info
        }

    def _extract_error_patterns(self, execution) -> List[Dict[str, Any]]:
        """Extract error patterns from execution"""
        patterns = []
        
        for event in execution.healing_events:
            pattern = {
                'pattern_type': 'error_recovery',
                'error_type': event['error_info'].get('error_type'),
                'task_type': event.get('task_type'),
                'retry_attempt': event['retry_attempt'],
                'context': {
                    'execution_time': execution.total_execution_time,
                    'task_count': len(execution.tasks),
                    'workflow_type': execution.metadata.get('type')
                }
            }
            patterns.append(pattern)
        
        return patterns

    async def _extract_workflow_patterns(self, execution) -> List[WorkflowPattern]:
        """Extract workflow patterns from successful execution"""
        patterns = []
        
        if execution.success_rate > 0.8:  # Only extract from successful workflows
            # Task sequence pattern
            task_sequence = [task.agent_type for task in execution.tasks]
            if len(task_sequence) > 1:
                pattern = WorkflowPattern(
                    pattern_id=f"sequence_{execution.id}",
                    pattern_type="task_sequence",
                    conditions={
                        'task_types': task_sequence,
                        'workflow_type': execution.metadata.get('type')
                    },
                    actions=[
                        {
                            'type': 'execute_sequence',
                            'sequence': task_sequence
                        }
                    ],
                    success_rate=execution.success_rate,
                    confidence=0.7,
                    examples=[execution.id]
                )
                patterns.append(pattern)
        
        return patterns

    def _pattern_matches_conditions(self, pattern: WorkflowPattern, conditions: Dict[str, Any]) -> bool:
        """Check if pattern matches given conditions"""
        for key, value in conditions.items():
            if key in pattern.conditions:
                if pattern.conditions[key] != value:
                    return False
        return True

    def _extract_text_for_embedding(self, content: Dict[str, Any]) -> str:
        """Extract meaningful text from content for embedding generation"""
        text_parts = []
        
        def extract_strings(obj, depth=0):
            if depth > 3:  # Prevent infinite recursion
                return
            
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    text_parts.append(str(key))
                    extract_strings(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_strings(item, depth + 1)
        
        extract_strings(content)
        return ' '.join(text_parts)

    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of data for storage"""
        if not data:
            return {}
        
        summary = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
            elif isinstance(value, (list, dict)):
                summary[f"{key}_type"] = type(value).__name__
                summary[f"{key}_size"] = len(value)
            else:
                summary[f"{key}_type"] = type(value).__name__
        
        return summary

    async def _consolidate_memories(self):
        """Consolidate memories by moving important short-term memories to long-term storage"""
        try:
            # Get short-term memories sorted by importance and access
            short_term_ids = self.memory_indices[MemoryType.SHORT_TERM]
            
            if len(short_term_ids) < self.consolidation_threshold:
                return
            
            # Calculate consolidation scores
            consolidation_candidates = []
            for memory_id in short_term_ids:
                memory = self.memories[memory_id]
                
                # Calculate consolidation score based on importance, access, and age
                age_factor = (time.time() - memory.created_at) / (24 * 3600)  # Days
                access_factor = memory.access_count / max(1, age_factor)
                
                consolidation_score = memory.importance_score * 0.4 + access_factor * 0.6
                
                consolidation_candidates.append((memory_id, consolidation_score))
            
            # Sort by consolidation score
            consolidation_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Move top candidates to long-term memory
            num_to_consolidate = min(len(consolidation_candidates) // 4, 100)
            
            for memory_id, score in consolidation_candidates[:num_to_consolidate]:
                memory = self.memories[memory_id]
                memory.memory_type = MemoryType.LONG_TERM
                memory.importance_score *= 1.1  # Boost importance for consolidated memories
                
                # Update indices
                self.memory_indices[MemoryType.SHORT_TERM].remove(memory_id)
                self.memory_indices[MemoryType.LONG_TERM].append(memory_id)
            
            # Remove least important short-term memories if still over threshold
            remaining_short_term = len(self.memory_indices[MemoryType.SHORT_TERM])
            if remaining_short_term > self.max_short_term_memories:
                to_remove = remaining_short_term - self.max_short_term_memories
                
                # Remove least important memories
                least_important = consolidation_candidates[-to_remove:]
                for memory_id, _ in least_important:
                    await self._remove_memory(memory_id)
            
            self.logger.info(f"Consolidated {num_to_consolidate} memories to long-term storage")
            
        except Exception as e:
            self.logger.error(f"Memory consolidation failed: {str(e)}")

    async def _remove_memory(self, memory_id: str):
        """Remove a memory record from all storage"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # Remove from main storage
            del self.memories[memory_id]
            
            # Remove from indices
            if memory_id in self.memory_indices[memory.memory_type]:
                self.memory_indices[memory.memory_type].remove(memory_id)
            
            # Remove from semantic index (requires rebuilding index)
            # In production, consider using more efficient index removal
            if memory_id in self.semantic_id_map:
                idx = self.semantic_id_map.index(memory_id)
                self.semantic_id_map.remove(memory_id)
                # Note: FAISS doesn't support efficient single vector removal
                # In production, consider rebuilding index periodically

    def get_memory_size(self) -> int:
        """Get total number of memories stored"""
        return len(self.memories)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        stats = {
            'total_memories': len(self.memories),
            'by_type': {
                memory_type.value: len(self.memory_indices[memory_type])
                for memory_type in MemoryType
            },
            'total_patterns': len(self.patterns),
            'hit_rate': self.memory_hits / max(1, self.memory_hits + self.memory_misses),
            'avg_importance': sum(m.importance_score for m in self.memories.values()) / max(1, len(self.memories)),
            'semantic_index_size': self.semantic_index.ntotal
        }
        
        return stats

    async def clear_working_memory(self):
        """Clear working memory for fresh start"""
        working_memory_ids = self.memory_indices[MemoryType.WORKING].copy()
        for memory_id in working_memory_ids:
            await self._remove_memory(memory_id)
        
        self.working_memory.clear()
        self.logger.info("Working memory cleared") 