from datetime import datetime
from typing import List
from pydantic import BaseModel


class EnrollmentResponseModel(BaseModel):
    
    matricNumber: str
    embeddings: List[List[float]]
    # embeddings: List[float]
    confidenceScore: float
    model: str = "SFace-2021Dec"
    created_at: datetime
    updated_at: datetime



class EnrolledResponseModel(BaseModel):
    
    matricNumber: str
    enrolled: str = 'True'
    confidenceScore: float
    model: str = "SFace-2021Dec"
    created_at: datetime
    updated_at: datetime


class VerificationResponseModel(BaseModel):
    matricNumber: str
    verified: bool
    similarity: float
    threshold: float
    message: str
    