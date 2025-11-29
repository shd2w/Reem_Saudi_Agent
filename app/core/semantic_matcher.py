"""
Semantic Search with Cosine Similarity for Service Matching
============================================================
Professional implementation of semantic search using sentence embeddings.
Supports Arabic and English text with normalized cosine similarity.

Architecture:
1. Exact Match (fastest, 100% accurate)
2. Keyword Mapping/Synonyms (fast, handles known variations)
3. Fuzzy Matching (handles typos, ~90% accurate)
4. Cosine Similarity (slowest, handles semantic meaning) ← THIS MODULE
5. Graceful Failure (ask user to clarify)
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class SemanticMatch:
    """Result of semantic matching"""
    service: Dict[str, Any]
    similarity_score: float  # 0.0 to 1.0
    match_type: str  # "semantic_high", "semantic_medium", "semantic_low"
    
    def __repr__(self):
        return f"SemanticMatch(service='{self.service.get('name')}', score={self.similarity_score:.2%}, type={self.match_type})"


class SemanticServiceMatcher:
    """
    Semantic matcher using sentence embeddings and cosine similarity.
    
    This is the FINAL fallback layer when all other methods fail.
    It understands semantic meaning, not just keywords.
    
    Examples:
    - "إزالة الشعر" → matches "ليزر" (hair removal → laser)
    - "تجميل الوجه" → matches "فيلر" (face beautification → filler)
    - "نضارة البشرة" → matches "ميزو" (skin freshness → meso)
    """
    
    def __init__(self):
        """Initialize semantic matcher with sentence transformer."""
        self.model = None
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self._initialized = False
        
    def _ensure_model_loaded(self):
        """Lazy load the model only when needed (saves memory)."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"🧠 Loading semantic model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info("✅ Semantic model loaded successfully")
        except ImportError:
            logger.error("❌ sentence-transformers not installed! Run: pip install sentence-transformers")
            self._initialized = False
        except Exception as e:
            logger.error(f"❌ Failed to load semantic model: {e}")
            self._initialized = False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        if not text:
            return ""
        
        # Basic normalization
        text = text.strip().lower()
        
        # Remove extra spaces
        text = " ".join(text.split())
        
        return text
    
    def _compute_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Compute embeddings for a list of texts."""
        if not self._initialized or self.model is None:
            return None
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to compute embeddings: {e}")
            return None
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        Returns value between -1 (opposite) and 1 (identical).
        We normalize to 0-1 range for easier interpretation.
        """
        # Compute dot product
        dot_product = np.dot(embedding1, embedding2)
        
        # Compute magnitudes
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to 0-1 range (cosine is -1 to 1, we shift to 0 to 1)
        normalized_similarity = (similarity + 1) / 2
        
        return float(normalized_similarity)
    
    def find_semantic_matches(
        self,
        query: str,
        services: List[Dict[str, Any]],
        threshold_high: float = 0.75,  # 75%+ = high confidence
        threshold_medium: float = 0.60,  # 60-75% = medium confidence
        threshold_low: float = 0.50,  # 50-60% = low confidence (show with caution)
        max_results: int = 5
    ) -> List[SemanticMatch]:
        """
        Find services that semantically match the query.
        
        Args:
            query: User's search query (e.g., "إزالة الشعر من الجسم")
            services: List of service dictionaries with 'name' field
            threshold_high: Minimum score for high confidence match
            threshold_medium: Minimum score for medium confidence match
            threshold_low: Minimum score for low confidence match
            max_results: Maximum number of results to return
        
        Returns:
            List of SemanticMatch objects, sorted by similarity (highest first)
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        if not self._initialized or self.model is None:
            logger.warning("⚠️ Semantic model not available, skipping semantic search")
            return []
        
        if not services:
            return []
        
        # Normalize query
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return []
        
        logger.info(f"🧠 SEMANTIC SEARCH: Finding matches for '{query}'")
        logger.info(f"   Searching {len(services)} services with thresholds: high≥{threshold_high:.0%}, med≥{threshold_medium:.0%}, low≥{threshold_low:.0%}")
        
        # Prepare service texts
        service_texts = []
        valid_services = []
        
        for service in services:
            # Get service name (try multiple fields)
            service_name = (
                service.get("name") or 
                service.get("name_ar") or 
                service.get("nameAr") or 
                ""
            )
            
            if not service_name:
                continue
            
            # Optionally include description for better matching
            description = service.get("description") or service.get("description_ar") or ""
            
            # Combine name and description (weight name more heavily)
            if description:
                combined_text = f"{service_name} {service_name} {description}"  # Name twice for emphasis
            else:
                combined_text = service_name
            
            normalized_text = self._normalize_text(combined_text)
            
            if normalized_text:
                service_texts.append(normalized_text)
                valid_services.append(service)
        
        if not service_texts:
            logger.warning("⚠️ No valid service texts to search")
            return []
        
        # Compute embeddings
        try:
            # Encode query
            query_embedding = self._compute_embeddings([normalized_query])
            if query_embedding is None:
                return []
            
            # Encode all services
            service_embeddings = self._compute_embeddings(service_texts)
            if service_embeddings is None:
                return []
            
            # Compute similarities
            matches: List[SemanticMatch] = []
            
            for idx, service in enumerate(valid_services):
                similarity = self._cosine_similarity(query_embedding[0], service_embeddings[idx])
                
                # Determine match type based on threshold
                if similarity >= threshold_high:
                    match_type = "semantic_high"
                elif similarity >= threshold_medium:
                    match_type = "semantic_medium"
                elif similarity >= threshold_low:
                    match_type = "semantic_low"
                else:
                    continue  # Below threshold, skip
                
                matches.append(SemanticMatch(
                    service=service,
                    similarity_score=similarity,
                    match_type=match_type
                ))
                
                logger.info(f"   ✅ {match_type.upper()}: '{service.get('name')}' (ID: {service.get('id')}, score: {similarity:.2%})")
            
            # Sort by similarity (highest first)
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit results
            matches = matches[:max_results]
            
            logger.info(f"✅ Found {len(matches)} semantic matches")
            return matches
        
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}", exc_info=True)
            return []
    
    def get_best_match(
        self,
        query: str,
        services: List[Dict[str, Any]],
        min_confidence: float = 0.60  # Only return if ≥60% confident
    ) -> Optional[SemanticMatch]:
        """
        Get the single best semantic match, if confidence is high enough.
        
        Args:
            query: User's search query
            services: List of service dictionaries
            min_confidence: Minimum confidence to return a match
        
        Returns:
            Best match if found and confident enough, None otherwise
        """
        matches = self.find_semantic_matches(
            query=query,
            services=services,
            threshold_high=0.75,
            threshold_medium=0.60,
            threshold_low=min_confidence,
            max_results=1
        )
        
        if matches and matches[0].similarity_score >= min_confidence:
            return matches[0]
        
        return None


# Global singleton instance (lazy loaded)
_semantic_matcher: Optional[SemanticServiceMatcher] = None


def get_semantic_matcher() -> SemanticServiceMatcher:
    """Get or create the global semantic matcher instance."""
    global _semantic_matcher
    
    if _semantic_matcher is None:
        _semantic_matcher = SemanticServiceMatcher()
    
    return _semantic_matcher
