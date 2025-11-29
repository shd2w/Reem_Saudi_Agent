"""
Adaptive Confidence Threshold System
=====================================
Dynamically adjusts confidence thresholds based on classification accuracy metrics.
"""

from typing import Dict, Tuple
from loguru import logger
from collections import defaultdict
import time


class AdaptiveConfidenceManager:
    """
    Manages adaptive confidence thresholds based on real-world accuracy.
    
    Tracks:
    - Classification accuracy per intent
    - User corrections (when intent was wrong)
    - Historical performance
    
    Adjusts thresholds to:
    - Lower thresholds for consistently accurate intents
    - Raise thresholds for frequently misclassified intents
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdaptiveConfidenceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        # Base thresholds (starting point)
        self.base_thresholds = {
            "fast_path_exact": 0.95,      # Exact keyword match
            "fast_path_high": 0.92,       # Strong pattern match
            "fast_path_medium": 0.88,     # Medium confidence
            "llm_high": 0.85,             # LLM high confidence
            "llm_medium": 0.75,           # LLM medium confidence
            "llm_low": 0.65               # LLM low confidence (needs validation)
        }
        
        # Current adjusted thresholds
        self.current_thresholds = self.base_thresholds.copy()
        
        # Track accuracy per intent
        self.intent_stats = defaultdict(lambda: {
            "total": 0,
            "correct": 0,
            "accuracy": 1.0,
            "last_update": time.time()
        })
        
        # Track threshold adjustments
        self.adjustment_history = []
        
        # Configuration
        self.min_samples = 10  # Minimum samples before adjusting
        self.adjustment_rate = 0.02  # How much to adjust per update
        self.max_adjustment = 0.15  # Maximum deviation from base
        
        self._initialized = True
        logger.info("✅ Adaptive Confidence Manager initialized")
    
    def get_threshold(self, threshold_type: str) -> float:
        """
        Get current threshold for a given type.
        
        Args:
            threshold_type: One of the threshold types (e.g., "fast_path_high")
            
        Returns:
            Current threshold value
        """
        return self.current_thresholds.get(threshold_type, 0.85)
    
    def record_classification(
        self,
        intent: str,
        confidence: float,
        was_correct: bool,
        threshold_type: str
    ):
        """
        Record a classification result to update accuracy metrics.
        
        Args:
            intent: Classified intent
            confidence: Confidence score
            was_correct: Whether the classification was correct
            threshold_type: Which threshold was used
        """
        stats = self.intent_stats[intent]
        stats["total"] += 1
        if was_correct:
            stats["correct"] += 1
        
        # Update accuracy
        stats["accuracy"] = stats["correct"] / stats["total"]
        stats["last_update"] = time.time()
        
        # Adjust threshold if we have enough samples
        if stats["total"] >= self.min_samples and stats["total"] % 10 == 0:
            self._adjust_threshold(intent, threshold_type, stats["accuracy"])
        
        logger.debug(
            f"📊 Intent stats: {intent} - "
            f"accuracy={stats['accuracy']:.2f} ({stats['correct']}/{stats['total']})"
        )
    
    def _adjust_threshold(self, intent: str, threshold_type: str, accuracy: float):
        """
        Adjust threshold based on accuracy.
        
        High accuracy (>0.95): Lower threshold (more lenient)
        Low accuracy (<0.80): Raise threshold (more strict)
        """
        base = self.base_thresholds.get(threshold_type, 0.85)
        current = self.current_thresholds.get(threshold_type, base)
        
        # Calculate adjustment
        if accuracy > 0.95:
            # High accuracy - can be more lenient
            adjustment = -self.adjustment_rate
            reason = "high_accuracy"
        elif accuracy < 0.80:
            # Low accuracy - need to be stricter
            adjustment = self.adjustment_rate
            reason = "low_accuracy"
        else:
            # Acceptable accuracy - no change
            return
        
        # Apply adjustment with bounds
        new_threshold = current + adjustment
        new_threshold = max(base - self.max_adjustment, min(base + self.max_adjustment, new_threshold))
        
        if new_threshold != current:
            self.current_thresholds[threshold_type] = new_threshold
            
            self.adjustment_history.append({
                "timestamp": time.time(),
                "intent": intent,
                "threshold_type": threshold_type,
                "old": current,
                "new": new_threshold,
                "accuracy": accuracy,
                "reason": reason
            })
            
            logger.info(
                f"🎯 THRESHOLD ADJUSTED: {threshold_type} "
                f"{current:.3f} → {new_threshold:.3f} "
                f"(intent={intent}, accuracy={accuracy:.2f}, reason={reason})"
            )
    
    def get_stats_summary(self) -> Dict:
        """Get summary of all intent statistics"""
        return {
            "intents": dict(self.intent_stats),
            "thresholds": self.current_thresholds,
            "adjustments": len(self.adjustment_history),
            "recent_adjustments": self.adjustment_history[-5:] if self.adjustment_history else []
        }
    
    def reset_to_base(self):
        """Reset all thresholds to base values"""
        self.current_thresholds = self.base_thresholds.copy()
        self.intent_stats.clear()
        self.adjustment_history.clear()
        logger.warning("⚠️ Reset all confidence thresholds to base values")


# Singleton instance
_confidence_manager = None

def get_confidence_manager() -> AdaptiveConfidenceManager:
    """Get or create the global confidence manager"""
    global _confidence_manager
    if _confidence_manager is None:
        _confidence_manager = AdaptiveConfidenceManager()
    return _confidence_manager
