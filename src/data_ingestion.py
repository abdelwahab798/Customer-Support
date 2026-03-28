import pandas as pd 
import logging 
import os 
import yaml
from sklearn.model_selection import train_test_split

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_ingestion.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
logger.addHandler(file_handler)


def load_data(file_path)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error occurred while loading data: {e}")
        raise

def simple_preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df=df.dropna()
        df=df.drop_duplicates()
        logger.info("Data simple preprocessed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error occurred while simple preprocessing data: {e}")
        raise

def edit_raws(df:pd.DataFrame)->pd.DataFrame:
    try:
        df["label"]=df["label"].str.replace("~", "")
        df["label"]=df["label"].str.replace("*", "")
        df["label"]=df["label"].str.lower()
        logger.info("Data raws edited successfully.")
        print(df.shape)
        return df 
    except Exception as e:
        logger.error(f"Error occurred while editing raws: {e}")
        raise
    
def save_data(df:pd.DataFrame, file_path:str):
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    raw_data_path=r"Data\Scraped_data\github_issues.csv"
    df=load_data(raw_data_path)
    df=simple_preprocess_data(df)
    df=edit_raws(df)
    save_data(df, r"Data\Processed_data\processed_github_issues.csv")

if __name__=="__main__": 
    main()