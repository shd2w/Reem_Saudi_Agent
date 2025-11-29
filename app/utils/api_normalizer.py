"""
API Response Normalizer
========================
Normalizes inconsistent API responses to a unified format.

Handles different response structures:
- {"data": [...]}
- {"results": [...], "count": 10}
- {"items": [...]}
- Direct arrays [...]

Author: Agent Orchestrator Team
Version: 1.0.0
"""

from typing import Dict, Any, List, Optional
from loguru import logger


def normalize_api_response(response: Any) -> Dict[str, Any]:
    """
    Normalize API response to consistent format.
    
    Args:
        response: Raw API response (dict, list, or None)
        
    Returns:
        Normalized response:
        {
            "items": [...],      # Always "items" key
            "count": int,        # Total count
            "success": bool,     # Success indicator
            "metadata": {...}    # Additional metadata
        }
    """
    try:
        # Handle None or empty response
        if not response:
            return {
                "items": [],
                "count": 0,
                "success": False,
                "metadata": {}
            }
        
        # Handle direct list response
        if isinstance(response, list):
            return {
                "items": response,
                "count": len(response),
                "success": True,
                "metadata": {}
            }
        
        # Handle dict response
        if isinstance(response, dict):
            # Check for "results" key (common format)
            if "results" in response:
                return {
                    "items": response["results"],
                    "count": response.get("count", len(response["results"])),
                    "success": True,
                    "metadata": {
                        k: v for k, v in response.items()
                        if k not in ["results", "count"]
                    }
                }
            
            # Check for "data" key (alternative format)
            elif "data" in response:
                data = response["data"]
                
                # If data is a list
                if isinstance(data, list):
                    return {
                        "items": data,
                        "count": len(data),
                        "success": True,
                        "metadata": {
                            k: v for k, v in response.items()
                            if k != "data"
                        }
                    }
                
                # If data is a dict (single item)
                elif isinstance(data, dict):
                    return {
                        "items": [data],
                        "count": 1,
                        "success": True,
                        "metadata": {
                            k: v for k, v in response.items()
                            if k != "data"
                        }
                    }
            
            # Check for "items" key (already normalized)
            elif "items" in response:
                return response
            
            # Single item response (no wrapper)
            else:
                return {
                    "items": [response],
                    "count": 1,
                    "success": True,
                    "metadata": {}
                }
        
        # Unknown format
        logger.warning(f"Unknown API response format: {type(response)}")
        return {
            "items": [],
            "count": 0,
            "success": False,
            "metadata": {"raw_response": str(response)}
        }
        
    except Exception as exc:
        logger.error(f"Error normalizing API response: {exc}")
        return {
            "items": [],
            "count": 0,
            "success": False,
            "metadata": {"error": str(exc)}
        }


def extract_items(response: Any) -> List[Dict]:
    """
    Quick helper to extract items from any response format.
    
    Args:
        response: Raw API response
        
    Returns:
        List of items (empty list if none found)
    """
    normalized = normalize_api_response(response)
    return normalized.get("items", [])


def get_item_count(response: Any) -> int:
    """
    Quick helper to get item count from any response format.
    
    Args:
        response: Raw API response
        
    Returns:
        Number of items
    """
    normalized = normalize_api_response(response)
    return normalized.get("count", 0)


def format_items_for_display(
    items: List[Dict],
    name_key: str = "name",
    max_items: int = 10,
    language: str = "arabic"
) -> str:
    """
    Format items as a bulleted list for display.
    
    Args:
        items: List of items to format
        name_key: Key to use for item name
        max_items: Maximum number of items to display
        language: Language for formatting
        
    Returns:
        Formatted string with bullet points
    """
    if not items:
        if language == "arabic":
            return "ما في نتائج متاحة حالياً"
        else:
            return "No results available"
    
    formatted_items = []
    for i, item in enumerate(items[:max_items], 1):
        name = item.get(name_key, "Unknown")
        
        # Add additional info if available
        additional_info = []
        if "type" in item:
            additional_info.append(item["type"])
        if "specialty" in item:
            additional_info.append(item["specialty"])
        
        if additional_info:
            formatted_items.append(f"• {name} ({', '.join(additional_info)})")
        else:
            formatted_items.append(f"• {name}")
    
    result = "\n".join(formatted_items)
    
    # Add "more items" indicator if truncated
    if len(items) > max_items:
        remaining = len(items) - max_items
        if language == "arabic":
            result += f"\n... و {remaining} خيارات أخرى"
        else:
            result += f"\n... and {remaining} more options"
    
    return result


# Example usage and tests
if __name__ == "__main__":
    # Test different response formats
    
    # Format 1: results + count
    response1 = {
        "results": [{"name": "Service 1"}, {"name": "Service 2"}],
        "count": 2
    }
    print("Format 1:", normalize_api_response(response1))
    
    # Format 2: data array
    response2 = {
        "data": [{"name": "Doctor 1"}, {"name": "Doctor 2"}]
    }
    print("Format 2:", normalize_api_response(response2))
    
    # Format 3: direct array
    response3 = [{"name": "Device 1"}, {"name": "Device 2"}]
    print("Format 3:", normalize_api_response(response3))
    
    # Format 4: single item
    response4 = {"name": "Patient 1", "id": 123}
    print("Format 4:", normalize_api_response(response4))
