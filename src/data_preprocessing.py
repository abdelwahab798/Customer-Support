import pandas as pd 
import logging 
import os 
from data_ingestion import simple_preprocess_data
from data_ingestion import simple_preprocess_data
from config import Config

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Data_preprocessing")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"Project.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
logger.addHandler(file_handler)

def label_mapping(label):
    if label in ["spam","invalid"]:
        return "Spam"
    elif label in ["bug","duplicate","caused-by-extension","freeze-slow-crash-leak","unit-test-failure"]:
        return "Bug"
    elif label in ["info-needed","question","author-verification-requested"]:
        return "Support"
    elif label in ["feature-request","enhancement","polish"]:
        return "Feature"
    else:
        return None
    

def features(df: pd.DataFrame,file_path:str) -> pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df["label"]=df["label"].map(label_mapping)
        logger.info("Features created successfully")
        df=simple_preprocess_data(df)
        logger.info("Data simple preprocessed successfully after creating features")
        return df
    except Exception as e:
        logger.error(f"Error occurred while creating features: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        file_path=Config.Data_processed_path
        df=features(df=None,file_path=file_path)
        save_data(df,Config.Data_full_processed_path)
    except Exception as e:
        logger.error(f"Error occurred in main function: {e}")
        raise

if __name__=="__main__":
    main()