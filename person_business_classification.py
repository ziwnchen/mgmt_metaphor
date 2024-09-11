import accelerate
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification # test model performance
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pickle
from  tqdm import tqdm

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# functions
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def load_model(save_directory):
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    model = BertForSequenceClassification.from_pretrained(save_directory)

    # If using GPU
    if torch.cuda.is_available():
        model = model.to('cuda')

    return tokenizer, model

def batch_predict(dataframe, tokenizer, model, batch_size=100):
    model.eval()
    predictions = []
    prediction_probs = []

    for start_idx in tqdm(range(0, len(dataframe), batch_size), desc="Processing"):
        # Process dataframe in smaller chunks
        end_idx = start_idx + batch_size
        text_chunk = dataframe['mgmt_sents'][start_idx:end_idx].tolist()

        # Tokenize the text chunk
        encodings = tokenizer(text_chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")

        # Create a custom dataset
        dataset = CustomDataset(encodings, [0]*len(text_chunk)) # Dummy labels
        dataloader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}

                # Get predictions
                outputs = model(**inputs)
                logits = outputs.logits

                # Convert logits to class probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=1)

                # Get the predicted class
                batch_predictions = torch.argmax(probabilities, dim=1)
                predictions.extend(batch_predictions.cpu().numpy())
                prediction_probs.extend(probabilities.cpu().numpy())

    return predictions, prediction_probs

# running
# load dataset 
# data_path = "/zfs/projects/faculty/amirgo-management/caselaw/processed/states/"
data_path = "/zfs/projects/faculty/amirgo-management/congress/speeches_processed/"
df = pd.read_pickle(data_path+f"total_mgmt_sent_tagged.pkl")

# select data to predict
verb_selected = df[(df['WSD_pred']==1)&(df['WSD_conf']>0.95)]
print(verb_selected.shape)
noun_selected = df[(df['noun_has_modifier']==True)]
print(noun_selected.shape)
df_test = pd.concat([verb_selected, noun_selected], ignore_index=True)

# Load your model and tokenizer
model_path = "/zfs/projects/faculty/amirgo-management/BERT/person_business_classifier_v2/"
tokenizer, model = load_model(model_path)
print("Tokenizer Ready!")

# # Run batch prediction
predictions, probabilities = batch_predict(df_test, tokenizer, model, batch_size=500)

# Add predictions and probabilities to DataFrame
df_test['pb_predictions'] = predictions
df_test['pb_probabilities'] = probabilities

# save
df_test.to_pickle(data_path+"person_business_classification.pkl")