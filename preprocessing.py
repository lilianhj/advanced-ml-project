from pathlib import Path
import numpy as np
import pandas as pd
import spacy
import torch
import torchtext
from torchtext.data import Field, TabularDataset, Dataset
import nltk
import scipy
import psycopg2 as ps
import dill


nlp = spacy.load("en")
nlp.max_length = 20000000


def connect_db():
    '''
    '''

    try:
        conn = ps.connect(host=host_name, database=dbname,
                        user=user_name, password=pwd, port=port)
    except ps.OperationalError as e:
        raise e
    else:
        print('Connected!')
    cur = conn.cursor()
    cur.execute(""" 
                SELECT * FROM raw_data; 
                """)
    data = cur.fetchall()
    colnames = [desc[0] for desc in cur.description] 
    conn.commit()

    #create the pandas dataframe
    df = pd.DataFrame(data)
    df.columns = colnames

    #close the connection
    cur.close()
    return df


def light_clean(df):
    '''
    '''
    sans_nulls_df = df[(df['alj_text'].notnull()) & (df['decision_binary'].notnull())]
    sans_nulls_df['alj_text'] = sans_nulls_df['alj_text']. \
                                    apply(lambda x: x[0: 1000000])
    return sans_nulls_df


def get_split_write(train_csv, val_csv, test_csv, split_yrs=(2017, 2019)):
    '''
    '''
    # get data
    df = connect_db()
    # clean it
    cleaned_df = light_clean(df)
    # split it by years & write out
    cleaned_df[cleaned_df['dab_year'] < split_yrs[0]].to_csv(train_csv, index=False)
    cleaned_df[(cleaned_df['dab_year'] >= split_yrs[0]) & \
            (cleaned_df['dab_year'] < split_yrs[1)].to_csv(val_csv, index=False)
    cleaned_df[cleaned_df['dab_year'] >= split_yrs[1].to_csv(test_csv, index=False)


def word_tokenize(text):
    tokenized = []
    # pass word list through language model.
    doc = nlp(text)
    for token in doc:
        if not token.is_punct and len(token.text.strip()) > 0:
            tokenized.append(token.text)
    return tokenized

def normalize_tokens(word_list, extra_stop=[]):
    #We can use a generator here as we just need to iterate over it
    normalized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list]) 
    doc = nlp(word_list.lower())
    # add the property of stop word to words considered as stop words
    if len(extra_stop) > 0:
        for stopword in extra_stop:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True
    for w in doc:
        # if it's not a stop word or punctuation mark, add it to our article
        if w.text != '\n' and not w.is_stop and not w.is_punct \
            and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
            normalized.append(str(w.lemma_))
    return normalized


def make_dataset(train_csv, val_csv):
    '''
    '''
    TEXT = Field(sequential=True, tokenize=word_tokenize,
                                 preprocessing=normalize_tokens)
    LABEL = Field(sequential=False)
    data_fields = [('alj_text', TEXT), ('decision_binary', LABEL)]
    train, val = TabularDataset.splits(path='', train=train_csv, validation=val_csv,
                                       format='csv', fields=data_fields, skip_header=True)
    return train, val, TEXT, LABEL


def other_shit(train, val, TEXT):
    TEXT.build_vocab(train, val)
    return TEXT

# from this 
# https://stackoverflow.com/questions/53421999/how-to-save-torchtext-dataset
def save_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path/"examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path/"fields.pkl", pickle_module=dill)


def load_dataset(path):
    if not isinstance(path, Path):
        path = Path(path)
    examples = torch.load(path/"examples.pkl", pickle_module=dill)
    fields = torch.load(path/"fields.pkl", pickle_module=dill)
    return Dataset(examples, fields)


def get_data