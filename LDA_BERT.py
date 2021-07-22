from abc import ABC, abstractmethod 
from Model import Model
import time
import datetime
import pandas as pd
import numpy as np
import torch
from transformers import *
import nltk
import spacy
import gensim
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

class LDA_BERT(Model):

    def __init__(self, epochs, train_dataloader, test_dataloader, orig_df):
        super().__init__(epochs, train_dataloader, test_dataloader)
        self.df = orig_df
        self.topic_vecs = self.topic_setup()

    def remove_stopwords(self, texts):
        sp = spacy.load('en_core_web_sm')
        stop_words = list(sp.Defaults.stop_words)
        stop_words.extend(['ll','ve','like','cause','got','gonna','wanna','yeah','said','oh','let','thing','time',
            'need','way','come','came','ain','gotta','away','don','need','ah','day','want','hey'])
        out = [[word for word in gensim.utils.simple_preprocess(str(doc))
            if word not in stop_words] 
            for doc in texts]
        return out

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def bigrams(self, words, bi_min=15):
        bigram = gensim.models.Phrases(words, min_count = bi_min)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod
        
    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def get_corpus(self):
        words = list(self.sent_to_words(self.df.lyrics_clean))
        words = self.remove_stopwords(words)
        bigram_mod = self.bigrams(words)
        bigram = [bigram_mod[review] for review in words]
        data_lemmatized = self.lemmatization(bigram)
        id2word = gensim.corpora.Dictionary(data_lemmatized)
        id2word.filter_extremes(no_below=10, no_above=0.3)
        id2word.compactify()
        corpus = [id2word.doc2bow(text) for text in data_lemmatized]
        return corpus, id2word, data_lemmatized

    def topic_setup(self):
        corpus, id2word, data_lemmatized = self.get_corpus()
        num_topics=7
        lda = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
            num_topics=num_topics, id2word=id2word, chunksize=100, workers=7,
            passes=50, eval_every = 1, per_word_topics=False)
        vecs = []
        for i in range(len(self.df)):
            top_topics = (lda.get_document_topics(corpus[i],minimum_probability=0.0))
            topic_vec = [top_topics[i][1] for i in range(num_topics)]
            vecs.append(topic_vec)
        return pd.DataFrame(vecs)

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
                loss_values.append(loss.item()/len(b_labels))
                # backward pass
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_loss / len(self.train_dataloader)            
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(self.format_time(time.time() - t0)))
            t0 = time.time()
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

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
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

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

        # extract info from hidden layers to supplement with spotify / LDA info
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
        
        train_y = train_labels.ravel()
        test_y = test_labels.ravel()

        pd.options.mode.chained_assignment = None

        # form final layers of model
        spot = self.df[['feature_instrumentalness', 'feature_tempo', 'feature_valence',
            'feature_energy', 'feature_speechiness', 'feature_acousticness',
            'feature_mode', 'feature_liveness', 'feature_loudness', 'feature_key',
            'feature_time_signature']].copy()

        transformer = Normalizer()
        spot = pd.DataFrame(transformer.transform(spot))
        train_spot, test_spot = train_test_split(spot, random_state=24, test_size=0.3)
        train_topics, test_topics = train_test_split(self.topic_vecs, random_state=24, test_size=0.3)

        train_spot.reset_index(inplace=True, drop=True)
        test_spot.reset_index(inplace=True, drop=True)
        train_topics.reset_index(inplace=True, drop=True)
        test_topics.reset_index(inplace=True, drop=True)

        train_features_all = pd.concat([pd.DataFrame(train_features), train_spot, train_topics], axis=1)
        test_features_all = pd.concat([pd.DataFrame(test_features), test_spot, test_topics], axis=1)

        train_features_all = train_features_all.copy().fillna(train_features_all.mean())
        test_features_all = test_features_all.copy().fillna(test_features_all.mean())

        export = False
        if export:
            features = pd.concat([train_features_all,test_features_all], axis=0)
            print(features.shape)
            features.reset_index(inplace=True, drop=True)
            full_df = pd.concat([self.df.reset_index(inplace=True, drop=True), features], axis=1)
            print(full_df.shape)
            full_df.to_csv('lda_layer_1.csv')

        nn = MLPClassifier(hidden_layer_sizes=(100,50), tol=0.007, early_stopping=True)
        nn.fit(train_features_all, train_y)
        score = nn.score(train_features_all, train_y)
        print("Train accuracy:", score)
        score = nn.score(test_features_all, test_y)
        print("Accuracy:", score)
        print("")

        print("Total time:", self.format_time(time.time() - t00))

        pred = nn.predict(test_features_all)
        df_pred = pd.DataFrame(pred)
        df_pred['actual'] = test_y
        df_pred.to_csv('pred3.csv')

        df_loss = pd.DataFrame(loss_values)
        df_loss.to_csv('loss_x.csv')
