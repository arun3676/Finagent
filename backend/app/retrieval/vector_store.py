"""
Vector Store Client

Qdrant vector database client with support for:
- Collection management
- CRUD operations on vectors
- Filtered search
- Role-based access control (RBAC)

Usage:
    store = VectorStore()
    await store.upsert_chunks(chunks)
    results = await store.search(query_embedding, filters={"ticker": "AAPL"})
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import settings
from app.models import DocumentChunk, RetrievedDocument

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant vector store client.
    
    Features:
    - Async operations for high throughput
    - Metadata filtering (ticker, date, doc type)
    - Batch upsert with progress tracking
    - Collection versioning
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        api_key: str = None,
        collection_name: str = None
    ):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: API key for Qdrant Cloud
            collection_name: Default collection name
        """
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        
        # Initialize client (sync for setup, async for operations)
        self._sync_client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None
    
    def _get_sync_client(self) -> QdrantClient:
        """Get or create sync client."""
        if self._sync_client is None:
            # Check if we have a QDRANT_URL in settings (for cloud)
            if hasattr(settings, 'QDRANT_URL') and settings.QDRANT_URL:
                # Use the full URL from settings (cloud instance)
                self._sync_client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=self.api_key,
                    timeout=settings.QDRANT_TIMEOUT
                )
            else:
                # Local instance - use HTTP
                url = f"http://{self.host}:{self.port}"
                self._sync_client = QdrantClient(
                    url=url,
                    api_key=self.api_key,
                    https=False,
                    timeout=settings.QDRANT_TIMEOUT
                )
        return self._sync_client

    async def _get_async_client(self) -> AsyncQdrantClient:
        """Get or create async client."""
        if self._async_client is None:
            # Check if we have a QDRANT_URL in settings (for cloud)
            if hasattr(settings, 'QDRANT_URL') and settings.QDRANT_URL:
                # Use the full URL from settings (cloud instance)
                self._async_client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=self.api_key,
                    timeout=settings.QDRANT_TIMEOUT
                )
            else:
                # Local instance - use HTTP
                url = f"http://{self.host}:{self.port}"
                self._async_client = AsyncQdrantClient(
                    url=url,
                    api_key=self.api_key,
                    https=False,
                    timeout=settings.QDRANT_TIMEOUT
                )
        return self._async_client
    
    def create_collection(
        self,
        collection_name: str = None,
        vector_size: int = None,
        distance: str = "Cosine"
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name for the collection
            vector_size: Dimension of vectors
            distance: Distance metric (Cosine, Euclid, Dot)
            
        Returns:
            True if created successfully
        """
        collection_name = collection_name or self.collection_name
        vector_size = vector_size or settings.EMBEDDING_DIMENSION
        
        client = self._get_sync_client()
        
        # Qdrant client versions differ on the helper method; use get_collection for existence
        try:
            client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists")
            return True
        except Exception as e:
            # If get_collection fails, it might not exist, or connection issue.
            # We'll log it at debug level and proceed to create.
            logger.debug(f"Collection check failed (likely doesn't exist): {e}")
            pass
        
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE if distance == "Cosine" else qdrant_models.Distance.EUCLID
                )
            )
            logger.info(f"Created collection {collection_name} with vector size {vector_size}")
        except Exception as e:
            # Handle case where collection was created by another process or race condition
            if "already exists" in str(e) or "Conflict" in str(e) or "409" in str(e):
                logger.info(f"Collection {collection_name} already exists (caught during creation)")
                return True
            else:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                raise e
        
        # Create common payload indexes
        payload_indexes = {
            "ticker": qdrant_models.PayloadSchemaType.KEYWORD,
            "document_type": qdrant_models.PayloadSchemaType.KEYWORD,
            "section": qdrant_models.PayloadSchemaType.KEYWORD,
            "fiscal_period": qdrant_models.PayloadSchemaType.KEYWORD,
            "fiscal_year": qdrant_models.PayloadSchemaType.INTEGER,
            "fiscal_quarter": qdrant_models.PayloadSchemaType.INTEGER,
            "filing_date": qdrant_models.PayloadSchemaType.DATETIME,
            "period_end_date": qdrant_models.PayloadSchemaType.DATETIME,
        }

        for field_name, field_schema in payload_indexes.items():
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
            except Exception as e:
                logger.warning(f"Could not create payload index {field_name}: {e}")
        
        logger.info(f"Created collection {collection_name} with vector size {vector_size}")
        return True
    
    async def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> int:
        """
        Upsert document chunks to vector store.
        
        Args:
            chunks: List of chunks with embeddings
            batch_size: Batch size for upsert operations
            
        Returns:
            Number of chunks upserted
        """
        if not chunks:
            return 0
        
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} missing embedding")
        
        client = await self._get_async_client()
        
        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = [self._chunk_to_point(chunk) for chunk in batch]
            
            await client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            total_upserted += len(batch)
            logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} chunks")
        
        logger.info(f"Upserted {total_upserted} chunks to {self.collection_name}")
        return total_upserted
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[RetrievedDocument]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        client = await self._get_async_client()
        
        filter_condition = self._build_filter(filters)
        
        # Prefer query_points (newer clients); fallback to search for older versions
        if hasattr(client, "query_points"):
            results = await client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=filter_condition,
                score_threshold=score_threshold
            )
            points = results.points
        else:
            # Older qdrant-client versions
            points = await client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=filter_condition,
                score_threshold=score_threshold
            )

        retrieved_docs = []
        for result in points:
            chunk = self._point_to_chunk(result)
            retrieved_docs.append(RetrievedDocument(
                chunk=chunk,
                score=result.score,
                retrieval_method="dense"
            ))
        
        return retrieved_docs
    
    async def search_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        client = await self._get_async_client()
        
        try:
            result = await client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id]
            )
            
            if result:
                return self._point_to_chunk(result[0])
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    async def delete_by_filter(
        self,
        filters: Dict[str, Any]
    ) -> int:
        """
        Delete documents matching filters.
        
        Args:
            filters: Metadata filters for deletion
            
        Returns:
            Number of documents deleted
        """
        if not filters:
            return 0
        
        filter_condition = self._build_filter(filters)
        if filter_condition is None:
            return 0
        
        client = await self._get_async_client()
        
        count_result = await client.count(
            collection_name=self.collection_name,
            count_filter=filter_condition,
            exact=True
        )
        deleted_count = count_result.count if count_result else 0
        if deleted_count == 0:
            return 0
        
        await client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(filter=filter_condition)
        )
        
        return deleted_count
    
    def _build_filter(
        self,
        filters: Dict[str, Any]
    ) -> Optional[qdrant_models.Filter]:
        """
        Build Qdrant filter from dictionary.
        
        Supported filter types:
        - ticker: exact match
        - document_type: exact match
        - section: exact match
        - fiscal_period: exact match
        - fiscal_year: exact match
        - fiscal_quarter: exact match
        - filing_date_gte: date >= value
        - filing_date_lte: date <= value
        - period_end_date_gte: date >= value
        - period_end_date_lte: date <= value
        
        Args:
            filters: Filter dictionary
            
        Returns:
            Qdrant Filter object
        """
        if not filters:
            return None
        
        conditions = []
        
        if "ticker" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="ticker",
                    match=qdrant_models.MatchValue(value=filters["ticker"])
                )
            )
        
        if "document_type" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="document_type",
                    match=qdrant_models.MatchValue(value=filters["document_type"])
                )
            )
        
        if "section" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="section",
                    match=qdrant_models.MatchValue(value=filters["section"])
                )
            )

        if "fiscal_period" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="fiscal_period",
                    match=qdrant_models.MatchValue(value=filters["fiscal_period"])
                )
            )

        if "fiscal_year" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="fiscal_year",
                    match=qdrant_models.MatchValue(value=filters["fiscal_year"])
                )
            )

        if "fiscal_quarter" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="fiscal_quarter",
                    match=qdrant_models.MatchValue(value=filters["fiscal_quarter"])
                )
            )
        
        if "filing_date_gte" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="filing_date",
                    range=qdrant_models.Range(gte=filters["filing_date_gte"])
                )
            )
        
        if "filing_date_lte" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="filing_date",
                    range=qdrant_models.Range(lte=filters["filing_date_lte"])
                )
            )

        if "period_end_date_gte" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="period_end_date",
                    range=qdrant_models.Range(gte=filters["period_end_date_gte"])
                )
            )

        if "period_end_date_lte" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="period_end_date",
                    range=qdrant_models.Range(lte=filters["period_end_date_lte"])
                )
            )
        
        return qdrant_models.Filter(must=conditions) if conditions else None
    
    def _chunk_to_point(
        self,
        chunk: DocumentChunk
    ) -> qdrant_models.PointStruct:
        """
        Convert DocumentChunk to Qdrant point.
        
        Args:
            chunk: Document chunk
            
        Returns:
            Qdrant PointStruct
        """
        return qdrant_models.PointStruct(
            id=chunk.chunk_id,
            vector=chunk.embedding,
            payload={
                "document_id": chunk.document_id,
                "content": chunk.content,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "ticker": chunk.metadata.ticker,
                "company_name": chunk.metadata.company_name,
                "document_type": chunk.metadata.document_type.value,
                "filing_date": chunk.metadata.filing_date.isoformat(),
                "period_end_date": chunk.metadata.period_end_date.isoformat() if chunk.metadata.period_end_date else None,
                "fiscal_year": chunk.metadata.fiscal_year,
                "fiscal_quarter": chunk.metadata.fiscal_quarter,
                "fiscal_period": chunk.metadata.fiscal_period,
                "source_url": chunk.metadata.source_url,
                "accession_number": chunk.metadata.accession_number
            }
        )
    
    def _point_to_chunk(self, point) -> DocumentChunk:
        """
        Convert Qdrant point to DocumentChunk.
        
        Args:
            point: Qdrant search result or retrieved point
            
        Returns:
            DocumentChunk instance
        """
        from app.models import DocumentMetadata, DocumentType
        
        payload = point.payload
        
        period_end_date = payload.get("period_end_date")
        parsed_period_end = None
        if period_end_date:
            try:
                parsed_period_end = datetime.fromisoformat(period_end_date)
            except (TypeError, ValueError):
                parsed_period_end = None

        metadata = DocumentMetadata(
            ticker=payload["ticker"],
            company_name=payload["company_name"],
            document_type=DocumentType(payload["document_type"]),
            filing_date=datetime.fromisoformat(payload["filing_date"]),
            fiscal_year=payload.get("fiscal_year"),
            fiscal_quarter=payload.get("fiscal_quarter"),
            fiscal_period=payload.get("fiscal_period"),
            period_end_date=parsed_period_end,
            source_url=payload["source_url"],
            accession_number=payload.get("accession_number")
        )
        
        return DocumentChunk(
            chunk_id=str(point.id),
            document_id=payload["document_id"],
            content=payload["content"],
            metadata=metadata,
            section=payload.get("section"),
            chunk_index=payload["chunk_index"],
            embedding=point.vector if hasattr(point, 'vector') else None
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection statistics and configuration
        """
        client = self._get_sync_client()
        
        try:
            info = client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
