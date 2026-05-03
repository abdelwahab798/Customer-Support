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
    Data_scraped_path = os.getenv("Data_scraped")
    Data_processed_path = os.getenv("Data_processed")
    Data_full_processed_path = os.getenv("Data_full_processed")
    Data_new_features_path = os.getenv("Data_new_feautres")
    Data_train_path = os.getenv("Data_train")
    Data_test_path = os.getenv("Data_test")
    Data_val_path = os.getenv("Data_val")

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