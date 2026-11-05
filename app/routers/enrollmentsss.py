import numpy as np
from datetime import datetime
from .. import models, schema
from .. database import enrollments
from .. util import extract_embedding, calculate_cosine_similarity
from fastapi import FastAPI, Response, status, HTTPException, APIRouter, File, Form, UploadFile


router = APIRouter(
    prefix="/enrollment",
    tags=['Enrollment']
)   

MATCH_THRESHOLD = 0.55


@router.post("/save", status_code=status.HTTP_201_CREATED, response_model=models.EnrolledResponseModel)
async def save_enrollment(
    matricNumber: str = Form(...),
    image: UploadFile = File(...)
):
    now = datetime.now()
    image_bytes = await image.read()
    result = extract_embedding(image_bytes)

    if result["success"] == False:
        error = result.get("error")

        if error == 'invalid_image':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Image")

        if error == 'multiple_faces_detected':
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Multiple Faces Detected")

        if error == 'no_face_detected':
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"No face detected from image")
    
    
    enrollments.update_one(
        { "matricNumber": matricNumber },
        {
            "$set": {
                "embeddings": [result["embedding"]],
                "confidenceScore": result["detection_confidence"],
                "model": "SFace-2021Dec",
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
            },
        },
        upsert=True,
    )
    
    document = enrollments.find_one({"matricNumber": matricNumber})

    return document


@router.get("/{matricNumber}", response_model=models.EnrollmentResponseModel)
def getEnrollment(matricNumber: str):

    document = enrollments.find_one({"matricNumber": matricNumber})

    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student with Matric Number: {matricNumber} was not found")
    
    if document["embeddings"] == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No enrollment record found for Student with Matric Number: {matricNumber}")


    return document
    # return [document, len(document["embeddings"][0])]


@router.post("/verify", status_code=status.HTTP_200_OK, response_model=models.VerificationResponseModel)
async def verify(
    matricNumber: str = Form(...),
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    MATCH_THRESHOLD = 0.55
    
    enrollment_record = enrollments.find_one(
        {"matricNumber": matricNumber},
        {"embeddings": 1, "_id": 0}
    )

    if not enrollment_record.get("embeddings"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No enrollment data found for Student with Matric Number: {matricNumber}")

    result = extract_embedding(image_bytes)
    
    if result["success"] == False:
        error = result.get("error")
    
        if error == 'invalid_image':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Image")
    
        if error == 'multiple_faces_detected':
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Multiple Faces Detected")
    
        if error == 'no_face_detected':
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"No face detected from image")

    current_embedding = result["embedding"]
    stored_embedding = enrollment_record["embeddings"]

    cosine_score = calculate_cosine_similarity(current_embedding, stored_embedding)


    if cosine_score >= MATCH_THRESHOLD:
        return {
            "matricNumber": matricNumber,
            "verified": True,
            "similarity": float(cosine_score),
            "threshold": MATCH_THRESHOLD,
            "message": "Face verified successfully"
        }
    else: 
        return {
            "matricNumber": matricNumber,
            "verified": False,
            "similarity": float(cosine_score),
            "threshold": MATCH_THRESHOLD,
            "message": "Face does not match the enrolled biometric record",
        }