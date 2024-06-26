import os
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import random
import torch
from torch.utils.data import DataLoader
import wandb

class allMiniLMModel:
    def __init__(self, model_name, num_labels, output_dir, train_dataset, valid_dataset, test_dataset, batch_size, epochs, learning_rate, seed, warmup_steps, weight_decay, wandb_project_name=None, wandb_api_key=None, wandb_entity=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.wandb_project_name = wandb_project_name
        self.warmup_steps = warmup_steps
        self.wandb_entity = wandb_entity
        self.weight_decay = weight_decay

          
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb
        if self.wandb_project_name:
            wandb.login()  # Utilisez la clé API de l'environnement ou celle passée directement
            wandb.init(project=self.wandb_project_name, entity=self.wandb_entity,
            config={
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "output_dir": self.output_dir,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "warmup_steps": self.warmup_steps
            })
        
        self.model = self._load_model()
        self.trainer = self._load_trainer()
        try:
            if self.epochs > 0:
                self.train()
        except Exception as e:
            print(f"Error occurred during training: {str(e)}")
        self.test()
    
    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        model.to(self.device)
        return model
    
    def _load_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=10,
            save_steps=100,
            save_total_limit=5,
            seed=self.seed,
            report_to="wandb"  # Ensure wandb is used for logging
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset
            
        )
        return trainer
    
    def _load_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def _compute_metrics(self, pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}
    
    def train(self):
        self.trainer.train()
    
    def test(self):
        result = self.trainer.evaluate(eval_dataset=self.test_dataset)
        if self.wandb_project_name:
            wandb.log(result)
        print("Test accuracy: ", result["eval_accuracy"])
