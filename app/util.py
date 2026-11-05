import os
import numpy as np
import cv2
from .config import MODEL_FILE_PATH, RECOGNIZER_FILE_PATH

from pathlib import Path

print("Model path:", MODEL_FILE_PATH)
print("Resolved path:", Path(MODEL_FILE_PATH).resolve())
print("Exists:", Path(MODEL_FILE_PATH).is_file())


if not MODEL_FILE_PATH.is_file():
    raise FileNotFoundError(f"YuNet model not found: {MODEL_FILE_PATH}")


detector = cv2.FaceDetectorYN.create(
        str(MODEL_FILE_PATH), "", 
        (320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )


recognizer = cv2.FaceRecognizerSF.create(
    model=RECOGNIZER_FILE_PATH, config=""
)


def extract_embedding(image_bytes):
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    

    if cv2_img is None:
        return {"success": False, "error": "invalid_image"}
    else:
        img_w = int(cv2_img.shape[1])
        img_h = int(cv2_img.shape[0])

    detector.setInputSize((img_w, img_h))


    count, faces = detector.detect(cv2_img)

    if faces is not None:
        if len(faces) > 1:
            return {"success": False, "error": "multiple_faces_detected"}
        else:
            alignedFace = recognizer.alignCrop(cv2_img, faces[0])
            faceFeature = recognizer.feature(alignedFace)
            featureToList = faceFeature.flatten().tolist()
            return {
                "success": True,
                "detection_confidence": f"{float(faces[0][-1]): .2f}",
                "embedding" :featureToList
                }
    else:
        return {"success": False, "error": "no_face_detected"}


def calculate_cosine_similarity(current_embedding, retrieved_embedding) -> float:
    current_feature = np.array(current_embedding, dtype=np.float32).reshape(1, 128)
    retrieved_feature = np.array(retrieved_embedding, dtype=np.float32).reshape(1, 128)
    cosine_score = recognizer.match(current_feature, retrieved_feature, cv2.FACE_RECOGNIZER_SF_FR_COSINE)

    return cosine_score



if __name__ == '__main__':
    with open("image2.jpeg", "rb") as file:
        image_bytes = file.read()
    
    result = extract_embedding(image_bytes)
    print(result)