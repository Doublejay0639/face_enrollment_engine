# receive image
# → decode it into an OpenCV image
# → detect exactly one face
# → align it
# → generate SFace embedding
# → convert NumPy embedding to list
# → save list in MongoDB

# import cv2
# import os


# model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx")


# detector = cv2.FaceDetectorYN.create(
#         model_path, "", 
#         (320, 320),
#         score_threshold=0.9,
#         nms_threshold=0.3,
#         top_k=5000
#     )

# print("YuNet:", MODEL_FILE_PATH)
# print("SFace:", RECOGNIZER_FILE_PATH)
# print(cv2.__version__)
# print(len(image_bytes))
# def receive_img(img):



import os
import numpy as np
import cv2
from config import MODEL_FILE_PATH, RECOGNIZER_FILE_PATH


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
                "detection_confidence": f"{float(faces[0][-1]): .6f}",
                "embedding" :featureToList
                }
    else:
        return {"success": False, "error": "no_face_detected"}


if __name__ == '__main__':
    with open("image2.jpeg", "rb") as file:
        image_bytes = file.read()
    
    result = extract_embedding(image_bytes)
    print(result)