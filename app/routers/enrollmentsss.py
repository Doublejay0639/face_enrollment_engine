import numpy as np
from datetime import datetime
from .. import models
from .. database import enrollments
from .. util import extract_embedding, calculate_cosine_similarity
from .. util2 import encrypt_embedding, decrypt_embedding
from fastapi import status, HTTPException, APIRouter, File, Form, UploadFile


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
    enrollment_record = enrollments.find_one({"matricNumber": matricNumber})
    
    if enrollment_record:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Student already has a biometric enrollment.")
    
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

        if error:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error extracting embedding from image")

    encrypted_embedding = encrypt_embedding(result["embedding"])
    
    write_result = enrollments.update_one(
        { "matricNumber": matricNumber },
        {
            "$setOnInsert": {
                "encrypted_embedding": encrypted_embedding,
                "confidenceScore": result["detection_confidence"],
                "model": "SFace-2021Dec",
                "updated_at": now,
                "created_at": now,
            },
        },
        upsert=True,
    )

    if write_result.upserted_id is None:
        raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Student already has a biometric enrollment."
    )
    
    document = enrollments.find_one({"matricNumber": matricNumber})

    return document


@router.put("/{matricNumber}", status_code=status.HTTP_200_OK, response_model=models.EnrolledResponseModel)
async def re_enroll(
    matricNumber: str,
    image: UploadFile = File(...)
):

    enrollment_record = enrollments.find_one({"matricNumber": matricNumber})
    
    if enrollment_record:
    
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

            if error:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error extracting embedding from image")


        encrypted_embedding = encrypt_embedding(result["embedding"])

        write_result = enrollments.update_one(
            { "matricNumber": matricNumber },
            {
                "$set": {
                    "encrypted_embedding": encrypted_embedding,
                    "confidenceScore": result["detection_confidence"],
                    "model": "SFace-2021Dec",
                    "updated_at": now,
                },
            }
        )

        if write_result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No existing record for student with matric number: {matricNumber}",
            )

                    
        document = enrollments.find_one({"matricNumber": matricNumber})        

        return document
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No existing record for student with matric number: {matricNumber}")



@router.get("/{matricNumber}", response_model=models.EnrollmentResponseModel)
def getEnrollment(matricNumber: str):

    document = enrollments.find_one({"matricNumber": matricNumber})

    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student with Matric Number: {matricNumber} was not found")
    
    if document.get("encrypted_embedding") == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No enrollment record found for Student with Matric Number: {matricNumber}")


    return document
   


@router.post("/verify", status_code=status.HTTP_200_OK)
async def verify(
    matricNumber: str = Form(...),
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    # MATCH_THRESHOLD = 0.55
    
    
    enrollment_record = enrollments.find_one(
        {"matricNumber": matricNumber},
        {"encrypted_embedding": 1, "_id": 0}
    )

    if not enrollment_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No enrollment data found for Student with Matric Number: {matricNumber}")

    if not enrollment_record.get("encrypted_embedding"):
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

        if error:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error extracting embedding from image")

    current_embedding = result["embedding"]
    encrypted_embedding_bytes = enrollment_record["encrypted_embedding"]

    try:
        decrypted_embedding_bytes = decrypt_embedding(encrypted_embedding_bytes)
        decrypted_embedding = np.frombuffer(decrypted_embedding_bytes, dtype=np.float32).reshape(1, 128)
    except ValueError as e:
        print("Authentication failed:", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server error")
        
    cosine_score = calculate_cosine_similarity(current_embedding, decrypted_embedding)


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