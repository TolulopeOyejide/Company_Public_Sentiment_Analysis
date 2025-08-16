from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    labels: List[str]
    probabilities: List[float]

class AnalyzeTwitterResponse(BaseModel):
    query: str
    total: int
    counts: Dict[str, int]
    examples: Dict[str, List[str]]
    results: List[Dict[str, Any]]
