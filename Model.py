from abc import ABC, abstractmethod
import time
import datetime
import numpy as np
import torch
from transformers import *

class Model(ABC):

    def __init__(self, epochs, train_dataloader, test_dataloader):
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model, self.optimizer, self.scheduler = self.model_setup()
        self.device = self.get_device()

    def model_setup(self):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
            num_labels = 2, output_attentions = False, output_hidden_states = True)
        model.cuda()
        optimizer = AdamW(model.parameters(),lr = 3e-5, eps = 1e-8)
        steps = len(self.train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps = 0, num_training_steps = steps)
        return model, optimizer, scheduler

    def get_device(self):
        if torch.cuda.is_available():      
            return torch.device("cuda")
        return torch.device("cpu")
    
    def compute_acc(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=rounded))

    def train(self):
        pass