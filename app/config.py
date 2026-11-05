import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")
# MODEL_DIR = BASE_DIR / "ml_models"



# MODEL_FILE = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

# RECOGNIZER_FILE = MODEL_DIR / "face_recognition_sface_2021dec.onnx"


MODEL_FILE = os.getenv("MODEL_PATH")
RECOGNIZER_FILE = os.getenv("RECOGNIZER_PATH")

MODEL_FILE_PATH = BASE_DIR / MODEL_FILE
RECOGNIZER_FILE_PATH = BASE_DIR / RECOGNIZER_FILE