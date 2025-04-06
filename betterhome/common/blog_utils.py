"""
Blog utilities for Better Home application.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

def is_how_to_query(query: str) -> bool:
    """
    Check if a query is a how-to question.
    
    Args:
        query: The user's query string
        
    Returns:
        bool: True if the query is a how-to question
    """
    print(f"Checking if '{query}' is a how-to query")
    patterns = [
        r"how\s+to",
        r"how\s+do\s+i",
        r"what\s+is\s+the\s+method\s+to",
        r"what\s+are\s+the\s+steps\s+to",
        r"can\s+you\s+explain\s+how\s+to",
        r"what\s+is\s+the\s+process\s+of",
        r"what\s+is\s+the\s+procedure\s+for",
        r"what\s+is\s+the\s+technique\s+for",
        r"what\s+is\s+the\s+approach\s+to",
        r"what\s+is\s+the\s+way\s+to"
    ]
    
    result = any(re.search(pattern, query.lower()) for pattern in patterns)
    print(f"Is how-to query: {result}")
    return result 