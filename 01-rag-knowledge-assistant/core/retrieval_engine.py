"""
Advanced RAG retrieval engine with robust error handling
Circuit breaker, retry logic, and comprehensive monitoring
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import traceback

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
except ImportError:
    FAISS = None
    HuggingFaceEmbeddings = None
    Document = None

from .document_processor import DocumentChunk, ProcessedDocument
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from simple_config import get_rag_config, get_monitoring_system

config = get_rag_config()
logger = logging.getLogger(__name__)
monitoring = get_monitoring_system("rag_retrieval")


class RetrievalError(Exception):
    """Base exception for retrieval operations"""
    pass

class EmbeddingError(RetrievalError):
    """Exception for embedding generation failures"""
    pass

class VectorStoreError(RetrievalError):
    """Exception for vector store operations"""
    pass

class QueryValidationError(RetrievalError):
    """Exception for invalid queries"""
    pass

class ModelLoadError(RetrievalError):
    """Exception for model loading failures"""
    pass

class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0

@dataclass
class RetrievalMetrics:
    """Retrieval operation metrics"""
    query_count: int = 0
    embedding_time: float = 0.0
    search_time: float = 0.0
    rerank_time: float = 0.0
    total_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

@dataclass 
class RetrievalResult:
    """Result of retrieval operation with metadata"""
    chunks: List[DocumentChunk]
    query: str
    similarity_scores: List[float]
    retrieval_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class CircuitBreaker:
    """Thread-safe circuit breaker pattern implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise RetrievalError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful operation"""
        async with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    async def _on_failure(self):
        """Handle failed operation"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )

class RetryHandler:
    """Async retry handler with exponential backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    async def execute(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    break
                
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                
                logger.warning(
                    f"Retry attempt {attempt + 1}/{self.config.max_attempts} failed",
                    extra={
                        "error": str(e),
                        "delay": delay,
                        "function": func.__name__
                    }
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception

class InputValidator:
    """Input validation for retrieval operations"""
    
    @staticmethod
    def validate_query(query: str) -> str:
        """Validate and sanitize query"""
        if not query or not isinstance(query, str):
            raise QueryValidationError("Query must be a non-empty string")
        
        query = query.strip()
        
        if len(query) < 3:
            raise QueryValidationError("Query must be at least 3 characters long")
        
        if len(query) > 1000:
            raise QueryValidationError("Query must be less than 1000 characters")
        
        # Remove potentially dangerous characters
        dangerous_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
        for char in dangerous_chars:
            query = query.replace(char, '')
        
        return query
    
    @staticmethod
    def validate_top_k(top_k: int) -> int:
        """Validate top_k parameter"""
        if not isinstance(top_k, int):
            raise QueryValidationError("top_k must be an integer")
        
        if top_k < 1:
            raise QueryValidationError("top_k must be at least 1")
        
        if top_k > 100:
            raise QueryValidationError("top_k must be at most 100")
        
        return top_k
    
    @staticmethod
    def validate_similarity_threshold(threshold: float) -> float:
        """Validate similarity threshold"""
        if not isinstance(threshold, (int, float)):
            raise QueryValidationError("Similarity threshold must be a number")
        
        if threshold < 0.0 or threshold > 1.0:
            raise QueryValidationError("Similarity threshold must be between 0.0 and 1.0")
        
        return float(threshold)

class EmbeddingGenerator:
    """Robust embedding generation with error handling"""
    
    def __init__(self):
        self.model = None
        self.model_name = config.embedding_model
        self._lock = asyncio.Lock()
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                expected_exception=EmbeddingError
            )
        )
        self.retry_handler = RetryHandler(
            RetryConfig(max_attempts=3, base_delay=1.0)
        )
        
    async def _load_model(self):
        """Load embedding model with error handling"""
        if self.model is not None:
            return
        
        try:
            monitoring.logger.info(
                "Loading embedding model",
                operation="load_embedding_model",
                model_name=self.model_name
            )
            
            start_time = time.time()
            
            # Use HuggingFace embeddings with error handling
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for stability
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Test the model with a simple query
            test_embedding = self.model.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ModelLoadError("Model loaded but returned empty embedding")
            
            load_time = time.time() - start_time
            
            monitoring.logger.info(
                "Embedding model loaded successfully",
                operation="load_embedding_model",
                model_name=self.model_name,
                load_time=load_time,
                embedding_dim=len(test_embedding)
            )
            
        except Exception as e:
            monitoring.logger.error(
                "Failed to load embedding model",
                operation="load_embedding_model",
                model_name=self.model_name,
                error=e
            )
            raise ModelLoadError(f"Failed to load embedding model: {str(e)}") from e
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with comprehensive error handling"""
        if not texts:
            return []
        
        # Validate inputs
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingError(f"Text at index {i} is not a string")
            if len(text.strip()) == 0:
                raise EmbeddingError(f"Text at index {i} is empty")
        
        async def _generate():
            async with self._lock:
                await self._load_model()
            
            try:
                start_time = time.time()
                
                # Generate embeddings in batches to avoid memory issues
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    try:
                        batch_embeddings = self.model.embed_documents(batch)
                        
                        # Validate embeddings
                        for j, embedding in enumerate(batch_embeddings):
                            if not embedding or len(embedding) == 0:
                                raise EmbeddingError(f"Empty embedding for text {i + j}")
                            
                            # Check for NaN or infinite values
                            if not all(np.isfinite(embedding)):
                                raise EmbeddingError(f"Invalid embedding values for text {i + j}")
                        
                        all_embeddings.extend(batch_embeddings)
                        
                    except Exception as e:
                        raise EmbeddingError(f"Failed to generate embeddings for batch {i//batch_size}: {str(e)}") from e
                
                generation_time = time.time() - start_time
                
                monitoring.logger.info(
                    "Generated embeddings successfully",
                    operation="generate_embeddings",
                    text_count=len(texts),
                    generation_time=generation_time,
                    avg_time_per_text=generation_time / len(texts)
                )
                
                return all_embeddings
                
            except Exception as e:
                monitoring.logger.error(
                    "Failed to generate embeddings",
                    operation="generate_embeddings",
                    text_count=len(texts),
                    error=e
                )
                raise EmbeddingError(f"Embedding generation failed: {str(e)}") from e
        
        return await self.circuit_breaker.call(
            self.retry_handler.execute,
            _generate
        )
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        if not query or not isinstance(query, str):
            raise EmbeddingError("Query must be a non-empty string")
        
        async def _generate():
            async with self._lock:
                await self._load_model()
            
            try:
                start_time = time.time()
                
                embedding = self.model.embed_query(query)
                
                if not embedding or len(embedding) == 0:
                    raise EmbeddingError("Generated empty embedding for query")
                
                if not all(np.isfinite(embedding)):
                    raise EmbeddingError("Generated invalid embedding values")
                
                generation_time = time.time() - start_time
                
                monitoring.logger.debug(
                    "Generated query embedding",
                    operation="generate_query_embedding",
                    query_length=len(query),
                    generation_time=generation_time
                )
                
                return embedding
                
            except Exception as e:
                monitoring.logger.error(
                    "Failed to generate query embedding",
                    operation="generate_query_embedding",
                    query=query[:100],  # Log first 100 chars
                    error=e
                )
                raise EmbeddingError(f"Query embedding generation failed: {str(e)}") from e
        
        return await self.circuit_breaker.call(
            self.retry_handler.execute,
            _generate
        )

class VectorStore:
    """Robust vector store with error handling and monitoring"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.index = None
        self.documents: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()
        self.is_built = False
        
    async def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to vector store with error handling"""
        if not documents:
            monitoring.logger.warning("No documents provided to add_documents")
            return
        
        try:
            monitoring.logger.info(
                "Adding documents to vector store",
                operation="add_documents",
                document_count=len(documents)
            )
            
            start_time = time.time()
            
            # Validate documents
            valid_documents = []
            for i, doc in enumerate(documents):
                try:
                    if not isinstance(doc, DocumentChunk):
                        raise VectorStoreError(f"Document {i} is not a DocumentChunk")
                    
                    if not doc.content or not doc.content.strip():
                        monitoring.logger.warning(f"Skipping empty document {i}")
                        continue
                    
                    # Ensure content is reasonable length
                    if len(doc.content) > 50000:  # 50k character limit
                        doc.content = doc.content[:50000]
                        monitoring.logger.warning(f"Truncated document {i} content")
                    
                    valid_documents.append(doc)
                    
                except Exception as e:
                    monitoring.logger.error(
                        f"Error validating document {i}",
                        operation="add_documents",
                        error=e
                    )
                    continue
            
            if not valid_documents:
                raise VectorStoreError("No valid documents to add")
            
            # Generate embeddings
            texts = [doc.content for doc in valid_documents]
            
            try:
                embeddings = await self.embedding_generator.generate_embeddings(texts)
            except Exception as e:
                raise VectorStoreError(f"Failed to generate embeddings: {str(e)}") from e
            
            # Add to store
            async with self._lock:
                self.documents.extend(valid_documents)
                
                if self.embeddings is None:
                    self.embeddings = np.array(embeddings, dtype=np.float32)
                else:
                    new_embeddings = np.array(embeddings, dtype=np.float32)
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
                
                # Rebuild FAISS index
                await self._build_index()
            
            add_time = time.time() - start_time
            
            monitoring.logger.info(
                "Documents added successfully",
                operation="add_documents",
                added_count=len(valid_documents),
                total_count=len(self.documents),
                add_time=add_time
            )
            
        except Exception as e:
            monitoring.logger.error(
                "Failed to add documents",
                operation="add_documents",
                error=e
            )
            raise VectorStoreError(f"Failed to add documents: {str(e)}") from e
    
    async def _build_index(self):
        """Build FAISS index with error handling"""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise VectorStoreError("No embeddings available to build index")
        
        try:
            dimension = self.embeddings.shape[1]
            
            # Use IndexFlatIP for inner product similarity
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = self.embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            self.index.add(embeddings_normalized)
            self.is_built = True
            
            monitoring.logger.info(
                "FAISS index built successfully",
                operation="build_index",
                dimension=dimension,
                document_count=len(self.documents)
            )
            
        except Exception as e:
            monitoring.logger.error(
                "Failed to build FAISS index",
                operation="build_index",
                error=e
            )
            raise VectorStoreError(f"Failed to build index: {str(e)}") from e
    
    async def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Tuple[List[DocumentChunk], List[float]]:
        """Search vector store with comprehensive error handling"""
        try:
            # Validate inputs
            query = InputValidator.validate_query(query)
            top_k = InputValidator.validate_top_k(top_k)
            similarity_threshold = InputValidator.validate_similarity_threshold(similarity_threshold)
            
            if not self.is_built or self.index is None:
                raise VectorStoreError("Vector store not built. Add documents first.")
            
            if len(self.documents) == 0:
                monitoring.logger.warning("No documents in vector store")
                return [], []
            
            start_time = time.time()
            
            # Generate query embedding
            try:
                query_embedding = await self.embedding_generator.generate_query_embedding(query)
                query_vector = np.array([query_embedding], dtype=np.float32)
                faiss.normalize_L2(query_vector)
            except Exception as e:
                raise VectorStoreError(f"Failed to generate query embedding: {str(e)}") from e
            
            # Search index
            try:
                search_k = min(top_k * 2, len(self.documents))  # Search more for filtering
                scores, indices = self.index.search(query_vector, search_k)
                
                scores = scores[0]  # Get first (and only) query results
                indices = indices[0]
                
            except Exception as e:
                raise VectorStoreError(f"FAISS search failed: {str(e)}") from e
            
            # Filter and prepare results
            results = []
            result_scores = []
            
            for score, idx in zip(scores, indices):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if idx >= len(self.documents):
                    monitoring.logger.warning(f"Invalid document index: {idx}")
                    continue
                
                if score < similarity_threshold:
                    continue
                
                results.append(self.documents[idx])
                result_scores.append(float(score))
                
                if len(results) >= top_k:
                    break
            
            search_time = time.time() - start_time
            
            monitoring.logger.info(
                "Vector search completed",
                operation="search",
                query=query[:100],
                results_found=len(results),
                search_time=search_time,
                top_k=top_k
            )
            
            return results, result_scores
            
        except Exception as e:
            monitoring.logger.error(
                "Vector search failed",
                operation="search",
                query=query[:100] if isinstance(query, str) else str(query),
                error=e
            )
            raise VectorStoreError(f"Search failed: {str(e)}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "document_count": len(self.documents),
            "is_built": self.is_built,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "index_type": type(self.index).__name__ if self.index else None
        }

class RetrievalEngine:
    """Main retrieval engine with comprehensive error handling"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(self.embedding_generator)
        self.metrics = RetrievalMetrics()
        self._lock = asyncio.Lock()
        
        monitoring.logger.info(
            "Retrieval engine initialized",
            operation="init"
        )
    
    async def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to the retrieval system"""
        try:
            await self.vector_store.add_documents(documents)
            
            monitoring.logger.info(
                "Documents added to retrieval engine",
                operation="add_documents",
                document_count=len(documents)
            )
            
        except Exception as e:
            monitoring.logger.error(
                "Failed to add documents to retrieval engine",
                operation="add_documents",
                error=e
            )
            raise
    
    async def retrieve(self, query: str, top_k: int = None, similarity_threshold: float = None) -> RetrievalResult:
        """Retrieve relevant documents with comprehensive error handling"""
        start_time = time.time()
        
        # Use config defaults if not provided
        top_k = top_k if top_k is not None else config.retrieval_top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else config.similarity_threshold
        
        try:
            # Validate inputs
            query = InputValidator.validate_query(query)
            top_k = InputValidator.validate_top_k(top_k)
            similarity_threshold = InputValidator.validate_similarity_threshold(similarity_threshold)
            
            monitoring.logger.info(
                "Starting document retrieval",
                operation="retrieve",
                query=query[:100],
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # Perform search
            chunks, scores = await self.vector_store.search(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            retrieval_time = time.time() - start_time
            
            # Update metrics
            async with self._lock:
                self.metrics.query_count += 1
                self.metrics.total_time += retrieval_time
            
            # Create result
            result = RetrievalResult(
                chunks=chunks,
                query=query,
                similarity_scores=scores,
                retrieval_time=retrieval_time,
                metadata={
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "vector_store_stats": self.vector_store.get_stats()
                }
            )
            
            monitoring.logger.info(
                "Document retrieval completed",
                operation="retrieve",
                query=query[:100],
                results_count=len(chunks),
                retrieval_time=retrieval_time,
                avg_score=sum(scores) / len(scores) if scores else 0
            )
            
            return result
            
        except Exception as e:
            retrieval_time = time.time() - start_time
            
            # Update error metrics
            async with self._lock:
                self.metrics.error_count += 1
                self.metrics.last_error = str(e)
            
            monitoring.logger.error(
                "Document retrieval failed",
                operation="retrieve",
                query=query[:100] if isinstance(query, str) else str(query),
                retrieval_time=retrieval_time,
                error=e
            )
            
            # Return error result instead of raising
            return RetrievalResult(
                chunks=[],
                query=query if isinstance(query, str) else str(query),
                similarity_scores=[],
                retrieval_time=retrieval_time,
                error=str(e)
            )
    
    async def batch_retrieve(self, queries: List[str], top_k: int = None, similarity_threshold: float = None) -> List[RetrievalResult]:
        """Retrieve documents for multiple queries with error isolation"""
        if not queries:
            return []
        
        monitoring.logger.info(
            "Starting batch retrieval",
            operation="batch_retrieve",
            query_count=len(queries)
        )
        
        results = []
        
        # Process queries with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent retrievals
        
        async def process_query(query: str) -> RetrievalResult:
            async with semaphore:
                return await self.retrieve(query, top_k, similarity_threshold)
        
        try:
            tasks = [process_query(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    monitoring.logger.error(
                        f"Batch retrieval failed for query {i}",
                        operation="batch_retrieve",
                        query=queries[i][:100],
                        error=result
                    )
                    
                    processed_results.append(RetrievalResult(
                        chunks=[],
                        query=queries[i],
                        similarity_scores=[],
                        retrieval_time=0.0,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            successful_count = sum(1 for r in processed_results if r.error is None)
            
            monitoring.logger.info(
                "Batch retrieval completed",
                operation="batch_retrieve",
                total_queries=len(queries),
                successful_queries=successful_count,
                failed_queries=len(queries) - successful_count
            )
            
            return processed_results
            
        except Exception as e:
            monitoring.logger.error(
                "Batch retrieval failed completely",
                operation="batch_retrieve",
                error=e
            )
            
            # Return error results for all queries
            return [
                RetrievalResult(
                    chunks=[],
                    query=query,
                    similarity_scores=[],
                    retrieval_time=0.0,
                    error=str(e)
                )
                for query in queries
            ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retrieval engine metrics"""
        return {
            "query_count": self.metrics.query_count,
            "error_count": self.metrics.error_count,
            "error_rate": self.metrics.error_count / max(self.metrics.query_count, 1),
            "avg_retrieval_time": self.metrics.total_time / max(self.metrics.query_count, 1),
            "last_error": self.metrics.last_error,
            "vector_store_stats": self.vector_store.get_stats()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on retrieval engine"""
        try:
            start_time = time.time()
            
            # Test embedding generation
            test_embedding = await self.embedding_generator.generate_query_embedding("test query")
            embedding_time = time.time() - start_time
            
            # Test vector store if built
            search_time = 0.0
            if self.vector_store.is_built:
                start_time = time.time()
                await self.vector_store.search("test query", top_k=1)
                search_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "embedding_model_loaded": self.embedding_generator.model is not None,
                "vector_store_built": self.vector_store.is_built,
                "document_count": len(self.vector_store.documents),
                "embedding_time": embedding_time,
                "search_time": search_time,
                "embedding_dimension": len(test_embedding),
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            monitoring.logger.error(
                "Health check failed",
                operation="health_check",
                error=e
            )
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.get_metrics()
            } 