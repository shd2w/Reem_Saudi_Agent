"""
Manual Review Queue API Routes
==============================
Endpoints for staff to manage failed registration queue.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.queue.manual_review_queue import get_manual_review_queue
from loguru import logger


router = APIRouter(prefix="/api/manual-review", tags=["Manual Review"])


class UpdateStatusRequest(BaseModel):
    """Request to update entry status."""
    status: str
    note: Optional[str] = None


@router.get("/pending")
async def get_pending_registrations(limit: int = 50):
    """
    Get all pending failed registrations for manual review.
    
    Returns:
        List of pending registration entries with user data and errors
    """
    try:
        queue = get_manual_review_queue()
        entries = queue.get_pending_entries(limit=limit)
        
        return {
            "status": "success",
            "count": len(entries),
            "entries": entries
        }
        
    except Exception as e:
        logger.error(f"❌ [MANUAL_REVIEW_API] Error getting pending entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_queue_stats():
    """
    Get manual review queue statistics.
    
    Returns:
        Stats on pending, reviewed, resolved, and cancelled entries
    """
    try:
        queue = get_manual_review_queue()
        stats = queue.get_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ [MANUAL_REVIEW_API] Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{entry_id}/status")
async def update_entry_status(entry_id: str, request: UpdateStatusRequest):
    """
    Update the status of a manual review entry.
    
    Args:
        entry_id: Entry identifier
        request: New status and optional note
        
    Returns:
        Success status
    """
    try:
        # Validate status
        valid_statuses = ["pending", "reviewed", "resolved", "cancelled"]
        if request.status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        
        queue = get_manual_review_queue()
        success = queue.update_entry_status(
            entry_id=entry_id,
            status=request.status,
            note=request.note
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
        
        return {
            "status": "success",
            "message": f"Entry {entry_id} updated to {request.status}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [MANUAL_REVIEW_API] Error updating entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{entry_id}")
async def get_entry(entry_id: str):
    """
    Get a specific manual review entry by ID.
    
    Args:
        entry_id: Entry identifier
        
    Returns:
        Entry details
    """
    try:
        import json
        from pathlib import Path
        
        queue = get_manual_review_queue()
        file_path = queue.queue_dir / f"{entry_id}.json"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
        
        with open(file_path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        
        return {
            "status": "success",
            "entry": entry
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [MANUAL_REVIEW_API] Error getting entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))
