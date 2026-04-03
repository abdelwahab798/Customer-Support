import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    Predictions_DIR = os.getenv("Predictions_DIR")

    NUM_LABELS = int(os.getenv("NUM_LABELS"))
    EPOCHS = int(os.getenv("EPOCHS"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

    ID2LABEL = {
        0: "Support",
        1: "Bug",
        2: "Spam",
        3: "Feature"
    }

    LABEL2ID = {
        "Support": 0,
        "Bug": 1,
        "Spam": 2,
        "Feature": 3
    }