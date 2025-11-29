"""
Semantic Service Matcher using OpenAI Embeddings

Provides intelligent service matching that understands:
- Synonyms (تنظيف البشرة → تقشير)
- Descriptions (ازالة التصبغات → تقشير)
- English (chemical peel → تقشير)
- Cross-language queries

Usage:
    matcher = SemanticServiceMatcher()
    await matcher.initialize_embeddings(services_list)
    matched_service, confidence = await matcher.find_best_match("تنظيف البشرة")
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    logger.warning("OpenAI not installed - semantic matching unavailable")


class SemanticServiceMatcher:
    """
    Semantic service matcher using OpenAI embeddings.
    
    Features:
    - Pre-computes embeddings for all services
    - Caches embeddings for 1 hour
    - Cosine similarity for matching
    - Fallback to keyword matching
    - Multi-language support
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_duration_minutes: int = 60):
        """
        Initialize semantic matcher.
        
        Args:
            api_key: OpenAI API key (optional, uses env var if not provided)
            cache_duration_minutes: How long to cache embeddings (default: 60 min)
        """
        if AsyncOpenAI is None:
            self.client = None
            logger.warning("⚠️ Semantic matching disabled - OpenAI not available")
            return
        
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        self.service_embeddings: Dict[int, Dict] = {}
        self.last_initialized: Optional[datetime] = None
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.model = "text-embedding-3-small"  # Cost-effective model
        
        logger.info(f"✨ SemanticServiceMatcher initialized (cache: {cache_duration_minutes} min)")
    
    @property
    def is_initialized(self) -> bool:
        """Check if embeddings are initialized and not expired"""
        if not self.service_embeddings or self.last_initialized is None:
            return False
        return datetime.now() - self.last_initialized < self.cache_duration
    
    @property
    def cache_age_seconds(self) -> int:
        """Get cache age in seconds"""
        if self.last_initialized is None:
            return -1
        return int((datetime.now() - self.last_initialized).total_seconds())
    
    async def initialize_embeddings(self, services: List[Dict], force_refresh: bool = False):
        """
        Pre-compute embeddings for all services.
        
        Args:
            services: List of service dictionaries
            force_refresh: Force refresh even if cache is valid
        """
        if self.client is None:
            logger.warning("⚠️ Cannot initialize - OpenAI client not available")
            return
        
        # Check if cache is still valid
        if not force_refresh and self.is_initialized:
            logger.info(f"✅ Using cached embeddings ({len(self.service_embeddings)} services) - age: {self.cache_age_seconds}s")
            return
        
        logger.info(f"🔍 Initializing embeddings for {len(services)} services...")
        
        # Prepare texts for embedding
        texts_to_embed = []
        service_ids = []
        
        for service in services:
            # Combine Arabic name, English name, and description
            name_ar = service.get('name_ar') or service.get('nameAr') or ''
            name_en = service.get('name') or service.get('name_en') or ''
            description = service.get('description') or service.get('description_ar') or ''
            
            # Create rich text representation
            service_text = f"{name_ar} {name_en} {description}".strip()
            
            if service_text:
                texts_to_embed.append(service_text)
                service_ids.append(service.get('id'))
        
        if not texts_to_embed:
            logger.warning("⚠️ No valid service texts to embed")
            return
        
        try:
            # Batch embed all services (more efficient)
            logger.debug(f"📤 Sending {len(texts_to_embed)} texts to OpenAI for embedding...")
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts_to_embed
            )
            
            # Store embeddings
            self.service_embeddings = {}
            for i, embedding_data in enumerate(response.data):
                service_id = service_ids[i]
                service = services[i]
                
                self.service_embeddings[service_id] = {
                    'embedding': embedding_data.embedding,
                    'service': service,
                    'name_ar': service.get('name_ar') or service.get('nameAr'),
                    'name_en': service.get('name') or service.get('name_en')
                }
            
            self.last_initialized = datetime.now()
            
            logger.info(f"✅ Embeddings initialized: {len(self.service_embeddings)} services")
            logger.info(f"💾 Next refresh at: {(self.last_initialized + self.cache_duration).strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            # Don't crash - fallback to keyword matching
    
    async def find_best_match(
        self,
        user_query: str,
        threshold: float = 0.65,
        top_k: int = 1
    ) -> Tuple[Optional[Dict], float]:
        """
        Find best matching service using semantic similarity.
        
        Args:
            user_query: User's search query (Arabic or English)
            threshold: Minimum similarity score (0.0-1.0)
            top_k: Return top K matches (default: 1)
        
        Returns:
            Tuple of (matched_service_dict, confidence_score)
            Returns (None, 0.0) if no match above threshold
        """
        if self.client is None or not self.is_initialized:
            logger.warning("⚠️ Semantic matching not available - use keyword fallback")
            return None, 0.0
        
        try:
            # Get query embedding
            logger.debug(f"🔍 Finding semantic match for: '{user_query}'")
            
            query_response = await self.client.embeddings.create(
                model=self.model,
                input=user_query
            )
            query_embedding = query_response.data[0].embedding
            
            # Calculate cosine similarity with all services
            similarities = []
            
            for service_id, data in self.service_embeddings.items():
                similarity = self._cosine_similarity(
                    query_embedding,
                    data['embedding']
                )
                
                if similarity >= threshold:
                    similarities.append({
                        'service_id': service_id,
                        'service': data['service'],
                        'name_ar': data['name_ar'],
                        'name_en': data['name_en'],
                        'similarity': similarity
                    })
            
            if not similarities:
                logger.info(f"❌ No semantic match found for '{user_query}' (threshold: {threshold})")
                return None, 0.0
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return best match
            best_match = similarities[0]
            
            logger.info(
                f"✅ Semantic match: '{user_query}' → {best_match['name_ar']} "
                f"(confidence: {best_match['similarity']:.2f})"
            )
            
            if len(similarities) > 1:
                other_matches = ', '.join([f"{s['name_ar']} ({s['similarity']:.2f})" for s in similarities[1:top_k]])
                logger.debug(f"   Other matches: {other_matches}")
            
            return best_match['service'], best_match['similarity']
            
        except Exception as e:
            logger.error(f"❌ Semantic matching error: {e}")
            return None, 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "initialized": self.is_initialized,
            "services_count": len(self.service_embeddings),
            "last_initialized": self.last_initialized.isoformat() if self.last_initialized else None,
            "age_seconds": self.cache_age_seconds,
            "cache_duration_minutes": self.cache_duration.total_seconds() / 60,
            "model": self.model
        }


# Singleton instance
_semantic_matcher: Optional[SemanticServiceMatcher] = None


def get_semantic_matcher() -> SemanticServiceMatcher:
    """Get singleton semantic matcher instance"""
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticServiceMatcher()
    return _semantic_matcher
