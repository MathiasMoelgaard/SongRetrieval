from abc import ABC, abstractmethod 
from Model import Model
import time
import datetime
import numpy as np
import pandas as pd
import torch
from transformers import *

class FT_BERT(Model):

    def train(self):
        t00 = time.time()
        loss_values = []
        for epoch_i in range(0, self.epochs):
            t0 = time.time()
            total_loss = 0
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                
                # move to gpu
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # clear previous gradients
                self.model.zero_grad()     

                # forward pass
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels.long())
                loss = outputs[0]
                total_loss += loss.item()

                # backward pass
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_loss / len(self.train_dataloader)            
            loss_values.append(avg_train_loss)
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(self.format_time(time.time() - t0)))

            t0 = time.time()
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            pred = []
            test_y = []
            for batch in self.test_dataloader:

                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():        
                    outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]

                # move back to cpu
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = self.compute_acc(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
                
                if len(pred) == 0:
                    pred = np.argmax(logits, axis=1).flatten()
                    test_y = label_ids.flatten()
                else:
                    pred = np.concatenate((pred, np.argmax(logits, axis=1).flatten()))
                    test_y = np.concatenate((test_y, label_ids.flatten()))
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

        print("Total time:", self.format_time(time.time() - t00))
        
        df_pred = pd.DataFrame(pred)
        df_pred['actual'] = test_y
        df_pred.to_csv('pred2.csv')
