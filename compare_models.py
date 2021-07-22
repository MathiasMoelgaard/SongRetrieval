import random
import pandas as pd
import numpy as np
import torch
from transformers import *
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from BERT import BERT
from FT_BERT import FT_BERT
from LDA_BERT import LDA_BERT

def main():

    df = pd.read_csv('data/binary_data.csv', index_col=0)

    # use gpu when possible
    if torch.cuda.is_available():      
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # tokenize input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized = df['lyrics_clean'].apply((lambda x: tokenizer.encode(x, max_length=512,
        truncation=True, add_special_tokens=True)))

    # pad input and form attention mask
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    # train test split & convert to tensors
    labels = df.tag.values
    indices = np.arange(len(labels))
    labels = pd.get_dummies(labels, drop_first=True)
    train_features, test_features, train_labels, test_labels, idx_train, idx_test = train_test_split(input_ids,
                                                                    labels, indices, random_state=24, test_size=0.3)
    train_masks, test_masks, _, _ = train_test_split(attention_mask, labels, random_state=24, test_size=0.3)

    train_features = train_features.clone().detach()
    test_features = test_features.clone().detach()

    train_labels = torch.tensor(train_labels.to_numpy())
    test_labels = torch.tensor(test_labels.to_numpy())

    train_masks = train_masks.clone().detach()
    test_masks = test_masks.clone().detach()

    # create data loaders
    batch_size = 8

    train_data = TensorDataset(train_features, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_features, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    epochs = 3


    # model 1: BERT out of the box
    print("\nTraining vanilla BERT...\n")
    model1 = BERT(epochs, train_dataloader, test_dataloader)
    model1.train()

    # model 2: BERT with fine-tuning
    print("\nTraining fine-tuned BERT...\n")
    model2 = FT_BERT(epochs, train_dataloader, test_dataloader)
    model2.train()

    # model 3: fine-tuned BERT with topic models
    print("\nTraining fine-tuned BERT + LDA...\n")
    model3 = LDA_BERT(epochs, train_dataloader, test_dataloader, df)
    model3.train()

if __name__ == "__main__":
    main()