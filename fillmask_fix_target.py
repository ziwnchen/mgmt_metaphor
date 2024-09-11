# similar to the fillmask_fix_target code

import accelerate
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax

import pandas as pd
import numpy as np
import pickle
from  tqdm import tqdm
import datetime

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def data_preprocessing(df, col, tokenizer, max_length):
    # selected_sentences = []
    selected_indices = []
    processed_data = []

    for idx, row in df.iterrows():
        sentence = row[col]
        inputs = tokenizer(sentence, max_length=max_length, truncation=True, return_tensors="pt")
        if tokenizer.mask_token_id in inputs['input_ids']:
            mask_indices = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist()
            processed_data.append({
                'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'mask_indices': mask_indices,
            })
            # selected_sentences.append(sentence)
            selected_indices.append(idx)

    return processed_data, selected_indices

class MaskedSentenceDataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': self.data[idx]['attention_mask'],
            'mask_indices': self.data[idx]['mask_indices'],
        }

# padding to align sentence tensor
def custom_collate_fn(batch):
    input_ids = [item['input_ids'].squeeze(0) for item in batch]  # Ensure each is 1D
    attention_masks = [item['attention_mask'].squeeze(0) for item in batch]  # Ensure each is 1D
    mask_indices = [item['mask_indices'] for item in batch]

    # # Debugging: Print shapes to verify
    # for i, ids in enumerate(input_ids):
    #     print(f"Input ID {i} shape: {ids.shape}")

    input_ids_padded = pad_sequence(input_ids, batch_first=True)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True)

    return {'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'mask_indices': mask_indices}

# return probability of certain tokens
def batch_predict_with_dataloader(model, dataset, tokenizer, target_entities, batch_size=100):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    model.eval()
    total_predictions = []


    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_indices = batch['mask_indices']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = softmax(outputs.logits, dim=-1)
            # print(predictions.shape)  # Debugging: Print the shape of the predictions tensor: should be [batch size, token length of padded sequence, vocabulary]
            for i in range(len(input_ids)):
                sequence_target_predictions = []
                for mask_index in mask_indices[i]:
                    masked_predictions = predictions[i, mask_index]
                    # select prediction of certain idx
                    target_predictions = masked_predictions[target_entities]
                    sequence_target_predictions.append(target_predictions)
                total_predictions.append(sequence_target_predictions)

    return total_predictions

def convert_to_df(predictions, selected_indices, df):
    # Initialize the new columns as lists
    df[f'object_target_pred_prob'] = [[] for _ in range(len(df))]

    for i in range(len(predictions)):
        idx = selected_indices[i]
        predicted_probs = predictions[i][0].tolist()
        df.at[idx, f'object_target_pred_prob'] = predicted_probs

    return df

# running
# load dataset
part = 1
data_path = '/zfs/projects/faculty/amirgo-management/opus/processed/'
df = pd.read_pickle(data_path+f"total_obj_sents_0729_{part}.pkl")

# Load your model and tokenizer
model_path = "/zfs/projects/faculty/amirgo-management/BERT/MacBERTh/"
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer and Model Ready!")

# load entity lists; convert to ids
def token_to_idx(token, tokenizer = tokenizer):
    return tokenizer.encode(token, add_special_tokens=False)
def idx_to_token(idx, tokenizer = tokenizer):
    return tokenizer.decode(idx)

def gen_token_idx(total_objects, tokenizer = tokenizer):
    filtered_objects = []
    bert_vocab = set(tokenizer.get_vocab().keys())
    for word in total_objects:
        if word in bert_vocab:
            filtered_objects.append(word)
    filtered_objects_idx = [token_to_idx(token)[0] for token in filtered_objects]
    return filtered_objects, filtered_objects_idx

with open(data_path + "human_nonhuman_masked_objects_0729.pkl", 'rb') as f:
    original_objects = pickle.load(f)

# a more ambiguous list
managing_objects = [
    'manager', 'employer', 'director', 'executive', 'chairman', 'leader','controller',  
    'investment', 'capital', 'budget', 'money', 'finance', 'fund', 'estate', 'property', 'equity', 'profit', 'margin',
    'revenue', 'income', 'salary', 'wage', 'pay', 'compensation', 'expense', 'cost', 'price', 'fee', 'charge','payment', 
    'bill', 'account', 'balance', 'credit', 'loan', 'mortgage', 'interest', 'tax', 'taxation', 'liability',
    'worker', 'labor', 'staff', 'personnel', 'subordinate', 'intern', 
    'business', 'commerce', 'corporation', 'firm', 'company', 'industry', 'market', 'economy', 'enterprise', 'trade', 'organization',
    'project', 'task', 'initiative', 'campaign', 'program', 
    'product', 'merchandise', 'commodity', 'goods', 'service', 'offering','brand','franchise', 'patent',
    'client', 'consumer', 'buyer', 'seller', 'consumption',
    'contract', 'agreement', 'deal', 'transaction', 'sale', 'purchase',
    'profession', 'job', 'occupation', 'career', 'position', 'duty', 'responsibility', 'obligation',
    'plan', 'solution', 'innovation', 'acquisition','negotiation', 'operation', 'production'
]

random_objects = ['adoption', 'aerial', 'agricultural', 'amtrak', 'announcements', 'antenna', 'brave', 'cadet', 'captures', 'carroll',
                   'champaign', 'charley', 'ecosystem', 'excuses', 'exit', 'french', 'freshman', 'goal', 'headache', 'inter', 'knock',
                     'liberty', 'lifeboat', 'london', 'manifest', 'mrs', 'multimedia', 'narcotics', 'nitrate', 'orr', 'ow', 'parliamentary', 
                     'plantation', 'proof', 'protect', 'provider', 'ready', 'reese', 'revolutionaries', 'ribbons', 'san', 'sanders', 
                     'satisfaction', 'scope', 'series', 'sucker', 'superstructure', 'whig', 'whiskey']

total_objects = managing_objects + random_objects + original_objects
total_objects, total_object_idx = gen_token_idx(total_objects) # filter out non-bert vocab

# Print set up (for sanity check)
today = datetime.date.today().strftime("%m%d")
data_name = data_path.split("/")[-3]
model_name = model_path.split("/")[-2]
total_target_num = len(total_object_idx)
print("Data: ", data_name)
print("Model: ", model_name)
print("Total Target: ", total_target_num)

## Run batch prediction
processed_data, selected_indices = data_preprocessing(df, 'object_mask', tokenizer, 128) # data processing, mostly truncation
masked_sentence_dataset = MaskedSentenceDataset(processed_data) # convert to a dataset
predictions = batch_predict_with_dataloader(model, masked_sentence_dataset, tokenizer, total_object_idx, batch_size=50) # run batch prediction
df_selected = convert_to_df(predictions, selected_indices, df) # format prediction to df

# save
df_selected.to_pickle(data_path+f"{part}_object_{total_target_num}_target_fillmask_{model_name}_{today}.pkl")