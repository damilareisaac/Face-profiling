from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FONT_SIZE = 0.5
STEP = 13
TEXT_THICKNESS = 1

IMAGE_DIR = BASE_DIR / "images_db"
MODEL_PATH = BASE_DIR / "model_weights"

AGE_MODEL_WEIGHT_PATH = MODEL_PATH / "age_model_weights.h5"
GENDER_MODEL_WEIGHT_PATH = MODEL_PATH / "gender_model_weights.h5"
RACE_MODEL_WEIGHT_PATH = MODEL_PATH / "race_model_single_batch.h5"
