import pandas as pd 
import logging 
import os 
from transformers import DistilBertTokenizerFast
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from torch import nn
import torch
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
from config import Config
import numpy as np

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


class CustomTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels= inputs.get("labels")
        outputs= model(**inputs)
        logits= outputs.logits
        loss_fct= nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _= precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def load_data(train:str,test:str,full_data:str)->tuple:
    try:
         df_train=pd.read_csv(train)
         df_test=pd.read_csv(test)
         df_full=pd.read_csv(full_data)
         logger.info(f"Data loaded successfully from {train}, {test}, and {full_data}")
         return df_train,df_test,df_full
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

def prepare_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame,df_val:pd.DataFrame, tokenizer):
    try:
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True)
        train_dataset= Dataset.from_pandas(df_train.reset_index(drop=True))
        test_dataset= Dataset.from_pandas(df_test.reset_index(drop=True))
        val_dataset= Dataset.from_pandas(df_val.reset_index(drop=True))
        train_dataset= train_dataset.map(tokenize, batched=True)
        test_dataset= test_dataset.map(tokenize, batched=True)
        val_dataset= val_dataset.map(tokenize, batched=True)

        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        logger.info(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Validation: {len(val_dataset)}")
        return train_dataset, test_dataset, val_dataset
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")
        raise

def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=Config.NUM_LABELS,
            id2label=Config.ID2LABEL,
            label2id=Config.LABEL2ID,
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def train(train_dataset, eval_dataset, model, tokenizer):
    try:
        training_args = TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            num_train_epochs=Config.EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=os.path.join(log_dir, "trainer_logs"),
            logging_steps=10,
        )

        trainer = CustomTrainer(
            class_weights=torch.tensor([4.5, 1.2, 1,6.5]),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")
        return trainer
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def save_metrics(trainer, test_dataset):
    try:
        logger.info("Starting predictions on test dataset...")
        preds_output = trainer.predict(test_dataset)
        logger.info("Predictions completed successfully")
        os.makedirs(Config.Predictions_DIR, exist_ok=True)
        with open(os.path.join(Config.Predictions_DIR, "test_metrics.json"), "w") as f:
            json.dump(preds_output.metrics, f, indent=4)
    except Exception as e:
        logger.error(f"Error during predictions: {e}")
        raise

def save_artifacts(trainer, tokenizer):
    try:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        trainer.save_model(Config.OUTPUT_DIR)
        tokenizer.save_pretrained(Config.OUTPUT_DIR)

        artifacts = {
            "id2label": Config.ID2LABEL,
            "label2id": Config.LABEL2ID,
            "model_name": Config.MODEL_NAME,
        }
        with open(os.path.join(Config.OUTPUT_DIR, "artifacts.json"), "w") as f:
            json.dump(artifacts, f, indent=4)

        logger.info(f"All artifacts saved to {Config.OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error saving artifacts: {e}")
        raise

def main():
    df_train,df_test,df_val= load_data(Config.Data_train_path,Config.Data_test_path,Config.Data_val_path)
    tokenizer= load_tokenizer()
    train_dataset, test_dataset, val_dataset= prepare_datasets(df_train,df_test,df_val,tokenizer)
    model= load_model()
    trainer= train(train_dataset, val_dataset, model, tokenizer)
    save_metrics(trainer, test_dataset)
    save_artifacts(trainer, tokenizer)

if __name__ == "__main__":
    main()