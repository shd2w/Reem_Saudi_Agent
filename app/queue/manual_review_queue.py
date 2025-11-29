"""
Manual Review Queue for Failed Registrations
===========================================
Stores failed registrations for manual processing by staff.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class ManualReviewQueue:
    """
    Queue system for failed registrations that need manual review.
    
    Features:
    - File-based persistent storage
    - JSON format for easy viewing
    - Automatic cleanup of old entries
    - Search and filter capabilities
    """
    
    def __init__(self, queue_dir: str = "data/manual_review"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Manual review queue initialized at {self.queue_dir}")
    
    def add_failed_registration(
        self,
        session_key: str,
        registration_data: Dict,
        error_message: str,
        retry_count: int = 0
    ) -> str:
        """
        Add a failed registration to the manual review queue.
        
        Args:
            session_key: User session identifier
            registration_data: User's registration data
            error_message: Error that caused failure
            retry_count: Number of retry attempts
            
        Returns:
            Queue entry ID
        """
        try:
            timestamp = datetime.now()
            entry_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{session_key.split(':')[-1]}"
            
            entry = {
                "id": entry_id,
                "session_key": session_key,
                "timestamp": timestamp.isoformat(),
                "status": "pending",  # pending, reviewed, resolved, cancelled
                "registration_data": registration_data,
                "error": error_message,
                "retry_count": retry_count,
                "notes": []
            }
            
            # Save to file
            file_path = self.queue_dir / f"{entry_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📋 [MANUAL_QUEUE] Added entry {entry_id} for review")
            logger.info(f"📋 [MANUAL_QUEUE] User: {registration_data.get('name')}, Phone: {registration_data.get('phone')}")
            logger.info(f"📋 [MANUAL_QUEUE] Error: {error_message[:100]}...")
            
            return entry_id
            
        except Exception as e:
            logger.error(f"❌ [MANUAL_QUEUE] Failed to add entry: {e}")
            return None
    
    def get_pending_entries(self, limit: int = 50) -> List[Dict]:
        """Get all pending manual review entries."""
        try:
            entries = []
            
            for file_path in self.queue_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                    
                    if entry.get("status") == "pending":
                        entries.append(entry)
                    
                    if len(entries) >= limit:
                        break
                        
                except Exception as e:
                    logger.error(f"❌ [MANUAL_QUEUE] Error reading {file_path}: {e}")
            
            # Sort by timestamp (newest first)
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return entries
            
        except Exception as e:
            logger.error(f"❌ [MANUAL_QUEUE] Error getting pending entries: {e}")
            return []
    
    def update_entry_status(
        self,
        entry_id: str,
        status: str,
        note: Optional[str] = None
    ) -> bool:
        """
        Update the status of a manual review entry.
        
        Args:
            entry_id: Entry identifier
            status: New status (reviewed, resolved, cancelled)
            note: Optional note to add
            
        Returns:
            True if successful
        """
        try:
            file_path = self.queue_dir / f"{entry_id}.json"
            
            if not file_path.exists():
                logger.error(f"❌ [MANUAL_QUEUE] Entry {entry_id} not found")
                return False
            
            # Load entry
            with open(file_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            
            # Update status
            entry["status"] = status
            entry["updated_at"] = datetime.now().isoformat()
            
            if note:
                entry["notes"].append({
                    "timestamp": datetime.now().isoformat(),
                    "note": note
                })
            
            # Save back
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ [MANUAL_QUEUE] Updated {entry_id} to status: {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [MANUAL_QUEUE] Error updating entry {entry_id}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get queue statistics."""
        try:
            stats = {
                "pending": 0,
                "reviewed": 0,
                "resolved": 0,
                "cancelled": 0,
                "total": 0
            }
            
            for file_path in self.queue_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                    
                    status = entry.get("status", "unknown")
                    stats[status] = stats.get(status, 0) + 1
                    stats["total"] += 1
                    
                except Exception:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ [MANUAL_QUEUE] Error getting stats: {e}")
            return {}


# Singleton instance
_queue_instance: Optional[ManualReviewQueue] = None


def get_manual_review_queue() -> ManualReviewQueue:
    """Get or create the manual review queue singleton."""
    global _queue_instance
    
    if _queue_instance is None:
        _queue_instance = ManualReviewQueue()
    
    return _queue_instance
