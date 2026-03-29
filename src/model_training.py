import pandas as pd 
import logging 
import os 
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from transformers import AutoTokenizer, DistilBertTokenizerFast
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from torch import nn
import torch
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Model_training")
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


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    try:
        classes= sorted(df["label"].unique())
        weights= compute_class_weight(class_weight="balanced",classes=np.array(classes),y=df["label"].values)
        logger.info(f"Class weights: {dict(zip(classes, weights))}")
        return torch.tensor(weights, dtype=torch.float)
    except Exception as e:
        logger.error(f"Error computing class weights: {e}")
        raise

class CustomTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels= inputs.pop("labels")
        outputs= model(**inputs)
        logits= outputs.logits
        loss= nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _= precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def load_data(train:str,test:str)->tuple:
    try:
         df_train=pd.read_csv(train)
         df_test=pd.read_csv(test)
         logger.info(f"Data loaded successfully from {train} and {test}")
         return df_train,df_test
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_tokenizer():
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def prepare_datasets(df: pd.DataFrame, tokenizer,train:str,test:str):
    try:
        df_train=pd.read_csv(train)
        df_test=pd.read_csv(test)
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True)
        train_dataset= Dataset.from_pandas(df_train.reset_index(drop=True))
        test_dataset= Dataset.from_pandas(df_test.reset_index(drop=True))
        train_dataset= train_dataset.map(tokenize, batched=True)
        test_dataset= test_dataset.map(tokenize, batched=True)
        
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        logger.info(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
        return train_dataset, test_dataset, df_train, df_test
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")
        raise