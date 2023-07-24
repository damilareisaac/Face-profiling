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

TONE_COLOURS = [
    "#373028",
    "#422811",
    "#513b2e",
    "#6f503c",
    "#81654f",
    "#9d7a54",
    "#bea07e",
    "#e5c8a6",
    "#e7c1b8",
    "#f3dad6",
    "#fbf2f3",
]
TONE_BLACK_WHITE = [
    "#FFFFFF",
    "#F0F0F0",
    "#E0E0E0",
    "#D0D0D0",
    "#C0C0C0",
    "#B0B0B0",
    "#A0A0A0",
    "#909090",
    "#808080",
    "#707070",
    "#606060",
    "#505050",
    "#404040",
    "#303030",
    "#202020",
    "#101010",
    "#000000",
]
