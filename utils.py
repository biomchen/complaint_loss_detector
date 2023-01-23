#!/usr/bin/env python3

import os
import re
import pandas as pd
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import multiprocessing as mp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class DataLoader:

    def choose_data(self, selection='all'):
        if selection not in ['all', 'bank', 'notnull']:
            print('Please select the correct data.')
            return
        cwd = os.getcwd()
        if selection == 'all':
            file_name = "complaints.csv"
        elif selection =='bank':
            file_name = "complaints_bank_all.csv"
        elif selection == 'notnull':
            file_name = "complaints_bank_notnull.csv"
        return os.path.join(cwd, file_name)

    def get_data(self, selection):
        data = self.choose_data(selection)
        df = pd.read_csv(data, low_memory=False)
        if 'Unnamed: 0' in df.columns:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)
        return df

    @property
    def get_columns(self):
        df = self.get_data()
        return df.columns

class DataProcessor:

    def transform(self, df):
        df = self.date_records(df)
        df = self.label_data(df)
        return df

    def date_records(self, df):
        df['Year'] = pd.to_datetime(df['Date received']).dt.year
        df['Month'] = pd.to_datetime(df['Date received']).dt.month
        df['Weekday'] = pd.to_datetime(df['Date received']).dt.weekday
        return df

    def label_data(self, df):

        # sorting data by checking if money relief is given based on classifier
        # binary: we will use the 1 for loss, 0 for not
        # multiclass: we will drop the "In progress" first, then label them with numbers

        # multi class dict
        class_dict = {
                'Closed with explanation': 0,
                'Closed with non-monetary relief': 0,
                'Closed without relief': 0,
                'Closed': 0,
                'Untimely response': 0,
                'Closed with relief': 1,
                'Closed with monetary relief': 2,
                'In progress': 3
        }
        # make binary loss feature
        df = df[df['Company response to consumer'].isin(class_dict)]
        df['Loss_binary'] = df['Company response to consumer'].map(
            lambda x:
            1 if x == 'Closed with monetary relief'
            else 0
        )
        # make multiclass loss feature
        df['Loss_multi'] = df['Company response to consumer'].map(
            class_dict
        )
        return df

class TextProcessor:

    lemma = WordNetLemmatizer()
    nlp = spacy.load("en_core_web_sm")
    swds = stopwords.words('english')

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        if type(X) == str:
            results = self.clean_text(X)
        else:
            pool = mp.Pool(mp.cpu_count())
            results = pool.map(self.clean_text, X)
            pool.close()
        return results

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_ners(self, x):
        ners = {}
        doc = TextProcessor.nlp(x)
        for ent in doc.ents:
            ners[ent.text] = ners.get(ent.text, 0) + 1
        return ners

    def clean_text(self, x):
        # removed punctuation
        x = re.sub(r'[^\w\s]', ' ', x)
        # remove number
        x = "".join(
            [char for char in x if char.isalpha() or char == " "]
        )
        # remove XXXX
        x = re.sub('[Xx|Yy|Zz]{2,}', '', x)
        # remove ner
        ners = self.get_ners(x)
        for ner in ners:
            x = re.sub(ner, '', x)
        # one space between words and remove stopwords
        x = " ".join(
            [
                word for word in x.lower().split()
                if word not in TextProcessor.swds
            ]
        )
        # tokenize
        x = nltk.word_tokenize(x)
        # lemmatization
        x = [TextProcessor.lemma.lemmatize(word, "v") for word in x]
        x = " ".join(x)
        return x

# call tfidf with customerized settings
tfidf = TfidfVectorizer(
    lowercase=False,
    max_df=0.3,
    min_df=10,
    ngram_range=(1, 3)
)

# build the ML pipeline
pipe = Pipeline(
    [
        ('tp', TextProcessor()),
        ('tfidf', tfidf),
        ('lg', LogisticRegression(
            class_weight='balanced', max_iter=10000, n_jobs=-1
            )
        )
    ]
)