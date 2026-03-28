import pandas as pd 
import logging 
import os 
from data_ingestion import simple_preprocess_data
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Feature_engineering")
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

def features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["label"]=df["label"].map(label_mapping)
        logger.info("Features created successfully")
        df=simple_preprocess_data(df)
        logger.info("Data simple preprocessed successfully after feature engineering")
        return df
    except Exception as e:
        logger.error(f"Error occurred while creating features: {e}")
        raise


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    try:
        label_map={"Support": 0, "Bug": 1, "Spam": 2,"Feature":3}
        df["label"]= df["label"].map(label_map)
        logger.info("Labels encoded successfully")
        return df
    except Exception as e:
        logger.error(f"Error occurred while encoding labels: {e}")
        raise

def concatenate_title_body(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["text"]=df["title"]+" "+df["body"]
        logger.info("Title and body concatenated successfully")
        df=simple_preprocess_data(df)
        logger.info("Data simple preprocessed successfully after concatenating title and body")
        df.drop(columns=["title","body"],inplace=True)
        logger.info("Dropped title and body columns successfully")
        return df
    except Exception as e:
        logger.error(f"Error occurred while concatenating title and body: {e}")
        raise

def save_data(df:pd.DataFrame, file_path:str):
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        df=pd.read_csv(r"Data\Processed_data\processed_github_issues.csv")
        df=features(df)
        df=encode_labels(df)
        df=concatenate_title_body(df)
        save_data(df,r"Data/Full_processed_data/feature_engineered_data.csv")
    except Exception as e:
        logger.error(f"Error occurred in main function: {e}")
        raise

if __name__=="__main__":
    main()