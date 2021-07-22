import sys
import re
import pickle
import numpy as np
import pandas as pd
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import random
from numba import jit, cuda
import matplotlib.pyplot as plt
import nltk

kmean = pd.read_csv("data/kmeans_groups.csv")
binary = pd.read_csv("data/binary_data.csv")

#Posting class is used from creation of inverted term document index, can be expanded to include emotional input but currently only based on text input. Self.position is currently not used but presented for if we want to consider positions of words and want to look for phrases rather than just the singular word input. One potential problem with this is limited data which is why it is currently not used as I judged that phrases will be very unique for a specific song. Though there are counter arguments to this.
#Self.docid - integer given to current song to later convert back to actual song. Saving as an integer enable us to save space as well as making the ranking process quicker and simpler.
#Self.tfidf - current song score on a given word(example in the bottom of this markdown)
#Self.tfidfReset - Used when generating cosine similarity
#Self.position - save location of word in text. For example in the text "Hello World" hello has position 0 and world has position 1
#Self.cosine - Compare ratios of words between songs, for example if a song has the word "fire" 3 times and "burn" 1 time and another song has "fire" 1 time and "burn" 1 time then it would compare those two vectors |xy|/|x||y|. The more words are in a query that are not present in a given song will bring this down highly.
#Example of small inverted index for the two songs "Love you baby" as song 1 and "My baby" is: {"my"} = {2} {"love"} = {1} {"you"} = {1} {"baby"} = {1,2} Each of those are a posting with docid 1 or 2 pointing back to the song
class Posting:
    def __init__(self, docid, tfidf, position):
        self.docid = docid
        self.tfidf = tfidf
        self.tfidfReset = 0
        self.position = position
        self.cosine = []
        
class retrivalModel:
    def __init__(self, sourceDataPath, rows=None, mode = 0, changeParams = [1,1,1,1,1,1,1,1,1]):
        self.inverseWIndex = {}
        self.nToHttp = {}
        self.features = {}
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.changeParams = changeParams
        self.kmean = pd.read_csv("data/kmeans_groups.csv")
        if (mode == 0):
            self.df = pd.read_csv(sourceDataPath, nrows=rows)
            self.dfTrue = self.df
            
    def getSource(self):
        return self.dfTrue
    
    def getInverse(self):
        return self.inverseWIndex
    
        
    #makes wordcount dict with frequency and locations of words in text
    def makedict(self, text, stem = True):
        count = 0
        mydict = {}
        words = self.tokenizer(text, stem)
        #print(words)
        #print(text)
        try:
            for word in words:
                count +=1
                if word not in mydict:
                    mydict[word] = [1, count]
                else:
                    mydict[word][0] += 1
                    mydict[word].append(count)#-mydict[word][len(mydict[word])-1])
            return mydict
        except:
            print("nonetype")
            return {}

    def setParams(self, params):
        self.changeParams = params
        
    #takes in text, returns words in stem that are lowercased and with special characters removed    
    def tokenizer(self, text, stem = True):
        ps = PorterStemmer() 
        try:
            if stem == False:
                return [word for word in re.findall(r'[a-z]+', text.lower())]
            return [ps.stem(word) for word in re.findall(r'[a-z]+', text.lower())]
        except:
            print(text)
            return []
    
    def songPrinter(self, query):
        for post in query:
            print(post.docid)
            print(post.tfidf)
            print(self.nToHttp[post.docid])
            
    def tf(self, n, size):
        return 100*n/size
    
    def tfidf(self):
        for k, v in self.inverseWIndex.items():
            for post in v:
                post.tfidf *= math.log(len(self.df.index)/(len(v) + 1), 10)
                post.tfidfReset = post.tfidf
                
    #takes in feature as tuple (str, int), mode as str
    def addFeature(self, feature, mode = "text"):
        if mode not in self.features:
            self.features[mode] = []
        self.features[mode].append(feature)
    
    #takes in mode and features to update relevant inverseIndex
    def inverse(self, mode, features, song, index):
        if mode == "text":
            for feature in features:
                #print(feature + " test")
                totalTerms = len(self.tokenizer(feature[0]))
                localDir = self.makedict(song[feature[0]])
                for k in localDir.keys():
                    if k not in self.inverseWIndex:
                        self.inverseWIndex[k] = []
                    self.inverseWIndex[k].append(Posting(index, feature[1]*self.tf(localDir[k][0], totalTerms), localDir[k][1:]))
    
    #created inverse index
    def createInverseIndex(self):
        try: 
            if self.features == {}:
                raise Exception("no features selected to build on")
            if self.inverseWIndex != {}:
                print("adding to previously created inverseIndex")
            for index, song in self.df.iterrows():
                self.nToHttp[index] = song["title"]
                #print(self.features)
                for k, v in self.features.items():
                    self.inverse(k, v, song, index)
            self.tfidf()
        except Exception as e:
            print(e)
    
    #Adds synonyms to the keywords extracted from given text
    def extend(self, word):
        synonyms = []
        i = 0
        for syn in wordnet.synsets(word):
            j = 0
            if i > int(self.changeParams[3]):
                break
            for l in syn.lemmas():
                if j > int(self.changeParams[2]):
                    break
                synonyms.append(l.name())
                #    synonyms.append(word)
                #    return synonyms 
                j +=1
            i += 1
        synonyms.append(word)
        return synonyms
    
    def prepBert(self, song):
        for k, v in self.features.items():
            #print(v)
            for item in v:
                if k == "text":
                    self.bertKeyWords(song[item[0]])
                    
    def queryDataPre(self, song): #query input should be keywords(word based/inverse index), mood(selection based), ...
        wordBaseQL = []
        for k, v in self.features.items():
            #print(v)
            for item in v:
                if k == "text":
                    wordBaseQL += self.extractKeyWords(song[item[0]])
        return wordBaseQL
                
    def extractKeyWords(self, text):
        ps = PorterStemmer() 
        localDir = self.makedict(text, False)
        if len(localDir) == 0:
            return []
        totalTerms = len(self.tokenizer(text))
        stop_words = set(stopwords.words('english'))
        stop_words.update(['t', 'd', 's', 'm', 'e', 're', 'll', 'il'])
        stop_words.update(['ll','ve','like','cause','got','gonna','wanna','yeah','said','oh', 'ooh', 'hm','let','thing','time',
            'need','way','come','came','ain','gotta','away','need','ah','day','want','hey'])
        tfidfScores = []
        nDoc = len(self.df.index)
        for k, v in localDir.items():
            tfScore = self.tf(v[0], totalTerms)
            if k not in stop_words and tfScore < 90:
                try:
                    #if wordnet.synsets(k)[0].pos() == "v":
            #if k not in stop_words and tfScore < 10:
                        tfidfScores.append((tfScore*math.log(nDoc/(len(self.inverseWIndex[ps.stem(k)])+1), 10), k))#10 self.changeParams[0]
                except:
                    pass
        tfidfScores.sort(key = lambda x: x[0], reverse = True)
        #return tfidfScores[:int(math.log(len(localDir), self.changeParams[0])) + int(self.changeParams[2])]
        tfidfWords = tfidfScores[:int(self.changeParams[4])]
        self.bertKeyWords(text)
        for word in self.bertWords:
            for compWord in tfidfScores:
                if word == compWord[1]:
                    tfidfWords.append(compWord)
                    break
        return tfidfWords
    
    def bertKeyWords(self, text):
        try:
            count = CountVectorizer(ngram_range=(1,1), stop_words="english").fit([text])
            candidates = count.get_feature_names()
            doc_embedding = self.model.encode([text])
            candidate_embeddings = self.model.encode(candidates)
            top_n = int(self.changeParams[1])
            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            self.bertWords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
        except:
            self.bertWords = []
        
        
    def query(self, song):
        mylist = []
        keyWords = self.queryDataPre(song)
        tfidf = []
        ps = PorterStemmer() 
        
        for key in keyWords:
            #try:
                keyExtend = self.extend(key[1])
                for k in set(keyExtend):
                    k = ps.stem(k)
                    try:
                        mylist.append(self.inverseWIndex[k])
                        tfidf.append(key[0])
                    except:
                        pass
            #except:
                #pass:
                #print(key[1] + " Not Found")
        
        try:
            query = mylist[0]
        except:
            return []
        
        for listing in mylist:
            #print("New listing")
            #listing.sort(key = lambda x: x.docid)
            for post in listing:
                #print(post.docid)
                post.tfidf = post.tfidfReset
                
        for i in range(0, len(query)):
            query[i].cosine = []
            query[i].cosine.append(query[i].tfidf)
            
        for i in range(1, len(mylist)): #
            query = self.ranker(query, mylist[i])
        #print(query)
        #print(len(mylist))
        query = self.cosine(query, tfidf, 80) #Top songs to return from regular rank
        query = self.bertRank(query, song, 20) #Top songs to return bertRank
        #self.songPrinter(query)
        return query
    
    def bertRank(self, query, song, topX):   
        embed = song.iloc[7:793]
        #songGroup = kmean.loc[num, 'Group']
        for post in query:
            querySong = self.getSource().iloc[post.docid]
            #queryGroup = kmean.loc[post.docid, 'Group']
            postEmbed = querySong.iloc[7:793]
            offSets = embed-postEmbed
            dis = np.dot(offSets.T, offSets) + .00001
            #print(dis)
            post.tfidf = self.changeParams[5]*post.tfidf + (1-self.changeParams[5])*post.tfidf/(np.sqrt(dis)*(self.changeParams[9])+.000001)
            #if songGroup == queryGroup:
            #    post.tfidf *= 10
        query.sort(key = lambda x: x.tfidf, reverse = True)
        return query[0:topX]
    
    def ranker(self, list1, list2):
        result = []
        p1 = 0
        p2 = 0
        len1 = len(list1)
        len2 = len(list2)
        high = 0
        while p1 < len1 and p2 < len2:
            list2[p2].cosine = []
            if list1[p1].docid < list2[p2].docid:
                #if (list1[p1].tfidf > 0*high):#increase 0 to prune documents in generated list so far with a low score
                list1[p1].cosine.append(0)
                high = max(high, len(list1[p1].cosine))
                result.append(list1[p1])
                p1 += 1
            elif list2[p2].docid < list1[p1].docid:
                #if (list2[p2].tfidf > 0*(high-1)):#increase 0 to prune documents in incoming list with a low score
                for i in range(0, high-1):
                    list2[p2].cosine.append(0)
                list2[p2].cosine.append(list2[p2].tfidf)
                result.append(list2[p2])
                p2 += 1
            else:
                list1[p1].cosine.append(list2[p2].tfidf)
                high = max(high, len(list1[p1].cosine))
                list1[p1].tfidf += list2[p2].tfidf
                list1[p1].tfidf *= self.changeParams[7]#0.8039956438026238 
                result.append(list1[p1])
                p1 += 1
                p2 += 1
        while p1 < len1:
            if list1[p1].docid < list2[p2-1].docid:
                print("fault")
            result.append(list1[p1])
            p1 +=1
        #while p2 < len2:
            #result.append(list2[p2])
            #p2 +=1
        return result
    
    def cosine(self, mylist, query, topX): #love:1,1 high:1,7
        for post in mylist:
            cos = self.calculateCosine(post.cosine, query)
            post.tfidf = self.changeParams[6]*post.tfidf + (1-self.changeParams[6])*post.tfidf*math.pow(cos, self.changeParams[8])
            #post.tfidf = post.tfidf + self.changeParams[3]*post.tfidf*math.pow(cos, self.changeParams[4])
        mylist.sort(key = lambda x: x.tfidf, reverse = True)
        return mylist[1:]
    
    def calculateCosine(self, list1, list2): #|yx|/|x|*|y|
        try:
            return np.dot(list1, list2)/(math.sqrt(np.dot(list1, list1)) * math.sqrt(np.dot(list2, list2)))
        except:
            return 0
    
    def categoricalEntropy(self, song, querySongList, trainParams):
        songMoods = song[trainParams]
        if querySongList == []:
            return 3
        myList = [0]*(len(trainParams))
        if (len(trainParams) == 1):
            myList.append(0)
        for querySong in querySongList:
            querySong = self.getSource().iloc[querySong.docid]
            moods = querySong[trainParams]
            if len(trainParams) == 1:
                for mood in moods:
                    myList[int(mood)] += 1
            else:
                maxFeature = 0
                j = 0
                for i, mood in enumerate(moods):
                    myList[i] += mood
        myList = [math.exp(i)/np.sum([math.exp(j) for j in myList]) for i in myList]
        if len(trainParams) == 1:
            for mood in songMoods:
                return -math.log(myList[int(mood)], 2)
        j = 0
        maxFeature = 0
        for i, mood in enumerate(songMoods):
            if mood > maxFeature:
                j = i
                maxFeature = mood
        return -math.log(myList[j], 2)#*25/len(querySongList) #+ .00004*sum([np.power(i, 2) for i in self.changeParams])
      
    def goldenFit(self, song, i, trainParams, a, b):
        p = .61803 #Golden ratio
        for _ in range(20):#increase this to try to narrow in on optimal point more
            #c and d and using golden ratio to try to narrow in on the optimal point
            d = p * b + (1-p) * a
            self.changeParams[i] = d
            #makes sure that all parameters that are used for logs stay within the domain range
            if (self.changeParams[i] < 2 and (i ==0 or i == 1)):
                self.changeParams[i] = 2
                #a = 2
            #makes sure that something can not be negatively relevant and that query return results
            if (self.changeParams[i] < 0):
                self.changeParams[i] = 0
                #a = 0
            if (self.changeParams[i] > 1 and (i == 5 or i == 6)):
                self.changeParams[i] = 1 
            querySongList = self.query(song)
            yd = self.categoricalEntropy(song, querySongList, trainParams)
            c = p * a + (1 - p) * b
            self.changeParams[i] = c
            if (self.changeParams[i] < 2 and (i ==0 or i == 1)):
                self.changeParams[i] = 2
                #b = 2
            if (self.changeParams[i] < 0):
                self.changeParams[i] = 0
                #b = 0
            if (self.changeParams[i] > 1 and (i == 5 or i == 6)):
                self.changeParams[i] = 1 
            querySongList = self.query(song)
            yc = self.categoricalEntropy(song, querySongList, trainParams)
            if yc < yd: #find lowest error between point c and d and narrow down scope to approach a
                b, d = d, c
            elif yc == yd: #will happen either when no query is returned or change doesn't affect query
                a += (b-a + .2)*(random.randint(0,200)/100 - 1)
                b += (b-a + .2)*(random.randint(0,200)/100 - 1)
            else: #narrow down scope to be closer to b
                a, c = c, d
        #print("a: {}, b: {}".format(a,b))
        self.changeParams[i] = c
        #makes sure that all parameters that are used for logs stay within the domain range
        if (self.changeParams[i] < 2 and (i ==0 or i == 1)):
            self.changeParams[i] = 2
            c = 2
        #makes sure that something can not be negatively relevant and that query return results
        if (self.changeParams[i] < 0):
            self.changeParams[i] = 0
            c = 0
        if (self.changeParams[i] > 1 and (i == 5 or i == 6)):
            self.changeParams[i] = 1 
        querySongList = self.query(song)
        yc = self.categoricalEntropy(song, querySongList, trainParams)
        
        self.changeParams[i] = d
        #makes sure that all parameters that are used for logs stay within the domain range
        if (self.changeParams[i] < 2 and (i ==0 or i == 1)):
            self.changeParams[i] = 2
            d = 2
        #makes sure that something can not be negatively relevant and that query return results
        if (self.changeParams[i] < 0):
            self.changeParams[i] = 0
            d = 0
        if (self.changeParams[i] > 1 and (i == 5 or i == 6)):
            self.changeParams[i] = 1    
        querySongList = self.query(song)
        yd = self.categoricalEntropy(song, querySongList, trainParams)
        if yc < yd:
            self.changeParams[i] = c
                
    def backtrackExp(self, song, i, trainParams, p):
        offSet = random.randint(0, int(p/10))/100 -.5*(p)/1000
        low = self.changeParams[i] - random.randint(0, p)/100 + offSet
        high = self.changeParams[i] + random.randint(0, p)/100 + offSet
        current = self.changeParams[i]
        querySongList = self.query(song)
        error = self.categoricalEntropy(song, querySongList, trainParams)
        self.goldenFit(song, i, trainParams, low, high)
        querySongList = self.query(song)
        newError = self.categoricalEntropy(song, querySongList, trainParams)
        #print("new: {}, old: {}".format(newError,error))
        #print("pastParam: {}, NewParam: {}".format(current,self.changeParams[i]))
        if error <= newError: #makes sure that nothing happens if error increased or stayed the same
            self.changeParams[i] = current
        
    def experimentalDescent(self, trainParams, changeParams, batch = 50):
        #based on quadratic fit rather than gradient
        self.changeParams = changeParams
        for i in range(len(changeParams)):
            if (self.changeParams[i] < 2 and (i ==0 or i == 1)):
                self.changeParams[i] = 2
            if (self.changeParams[i] < 0):
                self.changeParams[i] = 0
            if (self.changeParams[i] > 1 and (i == 5 or i == 6)):
                self.changeParams[i] = 1 
        errorTracker = []
        tempSongList = []
        changeUpdate = []
        bestError = 10000
        rate = 3000
        bestParams = changeParams.copy()
        #totalError = []
        count = 0
        self.df = shuffle(self.df)
        for _ in range(1):
            for index, song in self.df.iterrows():
                tempSongList.append(song)
                if len(tempSongList) == batch:
                    localChangeParam = self.changeParams.copy()
                    changeUpdate = [0]*len(self.changeParams)
                    for cSong in tempSongList:
                        #self.prepBert(cSong)
                        i = random.randint(0,len(changeParams)-1)
                        querySongList = self.query(cSong)
                        self.backtrackExp(song, i, trainParams, rate)
                        changeUpdate[i] += self.changeParams[i] - localChangeParam[i]
                        self.changeParams = localChangeParam.copy()
                    for i in range(len(self.changeParams)):
                        self.changeParams[i] += changeUpdate[i]/batch*len(changeParams)/2
                        if (self.changeParams[i] < 2 and (i ==0 or i == 1)):
                            self.changeParams[i] = 2
                        if (self.changeParams[i] < 0):
                            self.changeParams[i] = 0
                        if (self.changeParams[i] > 1 and (i == 5 or i == 6)):
                            self.changeParams[i] = 1
                        if (self.changeParams[i] > 2 and (i == 8 or i == 9)):
                            self.changeParams[i] = 2 
                    error = 0
                    for cSong in tempSongList:
                        querySongList = self.query(cSong)
                        error += self.categoricalEntropy(cSong, querySongList, trainParams)
                    if error < bestError:
                        bestParams = self.changeParams.copy()
                        bestError = error
                    tempSongList = []
                    rate = int(rate*.95)
                    if rate < 100:
                        rate = 100
                    print(self.categoricalEntropy(song, querySongList, trainParams))
                    print(self.changeParams)
                    errorTracker.append(error/batch)
                    count += 1
                    if count >= 6:
                        return errorTracker, bestParams
            #totalError.append(errorTracker)
            #errorTracker = []
        return errorTracker, bestParams

def binary_precision(runs, model, model2 = None):
    result = 0
    count = 0
    result2 = 0
    for _ in range(runs):
        num = random.randint(0,len(model.df.index)-1)
        song = model.getSource().iloc[num]
        targetTag = binary.loc[num, 'tag']
        for post in model.query(song):
            if binary.loc[post.docid, 'tag'] == targetTag:
                result +=1
            count += 1
        if model2 != None:
            for post in model2.query(song):
                if binary.loc[post.docid, 'tag'] == targetTag:
                    result2 +=1
    return result/count, result2/count