from abc import ABC, abstractmethod 
from Model import Model
import time
import datetime
import pandas as pd
import numpy as np
import torch
from transformers import *
from sklearn.neural_network import MLPClassifier

class BERT(Model):

    def train(self):
        t00 = time.time()
        self.model.eval()
        # pass train and test data through the model, storing the last hidden layer for each obs
        train_features = []
        train_labels = []
        for batch in self.train_dataloader:

            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():        
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            outputs = outputs.to_tuple()

            if len(train_features) == 0:
                train_features = outputs[1][-1][:,0,:].cpu().numpy()
                train_labels = b_labels.cpu().numpy()
            else:
                train_features = np.concatenate((train_features, outputs[1][-1][:,0,:].cpu().numpy()))
                train_labels = np.concatenate((train_labels, b_labels.cpu().numpy()))

        test_features = []
        test_labels = []
        for batch in self.test_dataloader:

            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():        
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            outputs = outputs.to_tuple()

            if len(test_features) == 0:
                test_features = outputs[1][-1][:,0,:].cpu().numpy()
                test_labels = b_labels.cpu().numpy()
            else:
                test_features = np.concatenate((test_features, outputs[1][-1][:,0,:].cpu().numpy()))
                test_labels = np.concatenate((test_labels, b_labels.cpu().numpy()))
        
        # using these as features, train our own final layers
        train_y = train_labels.ravel()
        test_y = test_labels.ravel()

        nn = MLPClassifier(hidden_layer_sizes=(80,50), max_iter=1000)
        nn.fit(train_features, train_y)
        score = nn.score(test_features, test_y)
        print("Accuracy:", score)
        print("")

        print("Total time:", self.format_time(time.time() - t00))

        pred = nn.predict(test_features)
        df_pred = pd.DataFrame(pred)
        df_pred['actual'] = test_y
        df_pred.to_csv('pred1.csv')
