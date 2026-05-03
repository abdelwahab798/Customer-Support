import pandas as pd 
import logging 
import os 
from data_ingestion import simple_preprocess_data
from sklearn.model_selection import train_test_split
from config import Config
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




def encode_labels(df: pd.DataFrame,file_path:str) -> pd.DataFrame:
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
        df.rename(columns={"label": "labels"}, inplace=True)
        logger.info("Title and body concatenated successfully and labels column renamed to labels")
        df=simple_preprocess_data(df)
        logger.info("Data simple preprocessed successfully after concatenating title and body")
        df.drop(columns=["title","body"],inplace=True)
        logger.info("Dropped title and body columns successfully")
        df["labels"]=df["labels"].astype(int)
        return df
    except Exception as e:
        logger.error(f"Error occurred while concatenating title and body: {e}")
        raise

def split(df:pd.DataFrame)->tuple:
    try:
        df_train, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["labels"])
        val_df, df_test = train_test_split(temp_df, test_size=0.5, stratify=temp_df["labels"], random_state=42)
        logger.info("Data split into train, validation, and test sets successfully")
        return df_train, df_test, val_df
    except Exception as e:
        logger.error(f"Error occurred while splitting data: {e}")
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
        file_path=Config.Data_processed_path
        df=pd.read_csv(file_path)
        df=encode_labels(df,file_path)
        df=concatenate_title_body(df)
        train_df,test_df,val_df=split(df)
        save_data(df,Config.Data_new_features_path)
        save_data(train_df,Config.Data_train_path)
        save_data(test_df,Config.Data_test_path)
        save_data(val_df,Config.Data_val_path)
    except Exception as e:
        logger.error(f"Error occurred in main function: {e}")
        raise

if __name__=="__main__":
    main()