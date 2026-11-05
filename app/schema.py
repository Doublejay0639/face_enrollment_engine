from typing import List, Optional
from pydantic import BaseModel


class EnrollmentSchema(BaseModel):
    
    matricNumber: str
    embeddings: List[List[float]]
    confidenceScore: float
    # embeddings: List[float]