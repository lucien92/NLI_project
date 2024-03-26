#%%
from datasets import load_dataset
import numpy as np
import pandas as pd
import random
import yaml
from model import allMiniLMModel
from data_utils import PreProcessor, Filter
from datasets import load_dataset

if __name__ == "__main__":
    
    snli = load_dataset("snli")

    train_dataset = load_dataset("snli", split='train')
    test_dataset = load_dataset("snli", split='test')
    validation_dataset = load_dataset("snli", split='validation')


    filter = Filter()
    preprocessor = PreProcessor()

    train_dataset_filtered = filter.transform(train_dataset)
    test_dataset_filtered = filter.transform(test_dataset)
    validation_dataset_filtered = filter.transform(validation_dataset)

    train_dataset_processed, labels_train = preprocessor.fit_transform(train_dataset_filtered) 
    test_dataset_processed, labels_test = preprocessor.transform(test_dataset_filtered)
    validation_dataset_processed, labels_val = preprocessor.transform(validation_dataset_filtered)

    config = yaml.safe_load(open("project/config.yml", "r"))
    training_config = config["training_config"]

    model_name=training_config["model_name"] 
    num_labels=training_config["num_labels"] 
    output_dir = training_config["output_dir"]

    batch_size=training_config["batch_size"] 
    epochs=training_config["epochs"] 
    learning_rate=training_config["learning_rate"] 
    seed=training_config["seed"] 
    warmup_steps=training_config["warmup_steps"] #grosses variations de learning rate au début
    #weight_decay=training_config["weight_decay"] #poids de la régularisation L2

    config_wandb = config["wandb"]
    wandb_project_name=config_wandb["project"]
    wandb_entity = config_wandb["entity"]

    model = allMiniLMModel(model_name, num_labels, output_dir, train_dataset_processed, validation_dataset_processed, test_dataset_processed, batch_size, epochs, learning_rate, seed, warmup_steps,wandb_project_name=None, wandb_api_key=None)