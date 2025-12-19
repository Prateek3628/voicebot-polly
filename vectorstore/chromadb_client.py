"""
ChromaDB client for vector storage and retrieval operations.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import config

logger = logging.getLogger(__name__)


class ChromaDBClient:
    """ChromaDB client for vector database operations."""
    
    def __init__(self):
        """Initialize ChromaDB client and embedding model."""
        try:
            # Store collection name
            self.collection_name = config.chromadb_collection_name
            
            # Ensure persist directory exists
            Path(config.chromadb_persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=config.chromadb_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=config.chromadb_collection_name
                )
                logger.info(f"Loaded existing collection: {config.chromadb_collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=config.chromadb_collection_name,
                    metadata={"description": "Privacy policy document chunks"}
                )
                logger.info(f"Created new collection: {config.chromadb_collection_name}")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def load_and_chunk_document_from_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Load and chunk a document from text content.
        
        Args:
            text: Document text content
            metadata: Document metadata
            
        Returns:
            List of document chunks
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            logger.info(f"Created {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to ChromaDB collection.
        
        Args:
            documents: List of document chunks to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in documents]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.get_collection_count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            logger.debug(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Document count
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=config.chromadb_collection_name)
            logger.info(f"Deleted collection: {config.chromadb_collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset the collection by deleting and recreating it.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing collection
            try:
                self.client.delete_collection(name=config.chromadb_collection_name)
                logger.info(f"Deleted existing collection: {config.chromadb_collection_name}")
            except Exception:
                pass  # Collection might not exist
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=config.chromadb_collection_name,
                metadata={"description": "Privacy policy document chunks"}
            )
            logger.info(f"Created new collection: {config.chromadb_collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents (Langchain compatible interface).
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        try:
            results = self.search_similar_documents(query, n_results=k)
            documents = []
            for result in results:
                documents.append(Document(
                    page_content=result['content'],
                    metadata=result['metadata']
                ))
            return documents
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information
        """
        try:
            count = self.get_collection_count()
            collection_info = {
                'name': config.chromadb_collection_name,
                'document_count': count,
                'embedding_model': config.embedding_model,
                'persist_directory': config.chromadb_persist_directory
            }
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def is_collection_empty(self) -> bool:
        """
        Check if the collection is empty.
        
        Returns:
            True if collection is empty, False otherwise
        """
        return self.get_collection_count() == 0
