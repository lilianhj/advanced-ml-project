'''
Functions to process the raw data to be ready for Pytorch

May/June 2020
'''
import pandas as pd
import spacy
import torch
from torchtext.data import Field, LabelField, TabularDataset
import psycopg2 as ps
import db_connection

NLP = spacy.load("en")
NLP.max_length = 20000000


def connect_db(host_name=db_connection.host_name, dbname=db_connection.dbname,
               user_name=db_connection.user_name, pwd=db_connection.pwd,
               port=db_connection.port):
    '''
    Connects to database and pulls all cases into a pandas df

    Inputs:
        host_name(str): name of host for db
        dbname(str): name of db
        user_name(str): db username
        pwd(str): password for db
        port(str): port for db

    Outputs: pandas dataframe of cases
    '''
    try:
        conn = ps.connect(host=host_name, database=dbname,
                          user=user_name, password=pwd, port=port)
    except ps.OperationalError as error:
        raise error
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
    all_df = pd.DataFrame(data)
    all_df.columns = colnames

    #close the connection
    cur.close()
    return all_df


def light_clean(raw_df):
    '''
    Lightly cleans the dataframe by removing obs with null values for
    necessary columns and cutting obs with alj_text greater than 1000000 words

    Inputs:
        raw_df: a pandas dataframe

    Output:
    '''
    sans_nulls_df = raw_df.loc[(raw_df['alj_text'].notnull()) & \
                               (raw_df['decision_binary'].notnull()),]
    sans_nulls_df['alj_text'] = sans_nulls_df['alj_text'].str.slice(0, 1000000)
    return sans_nulls_df


def get_split_write(train_csv, test_csv, val_csv, sample_size,
                    split_yrs=(2017, 2019)):
    '''
    Grabs data from db, cleans relevant cols, splits data into train/val/test
    datasets and writes these to csvs

    Inputs:
        train_csv(str): desired name of csv for training data
        test_csv(str): desired name of csv for testing data
        val_csv(str): desired name of csv for validation data
        sample_size(int): number of obs to grab if testing and want a
            small sample for speed
        split_yrs(tup): tuple of years that divide dataset from training
            to validation to testing

    Output: None (saves three csvs of train/val/test data)
    '''
    desired_cols = ['dab_id', 'alj_id', 'alj_text', 'decision_binary',
                    'dab_year']
    # get data
    raw_df = connect_db()
    # clean it
    cleaned_df = light_clean(raw_df)
    if not sample_size:
        use_df = cleaned_df
    else:
        use_df = cleaned_df.sample(sample_size, random_state=1312)
    # split it by years & write out
    use_df[use_df['dab_year'] < split_yrs[0]][desired_cols].to_csv(train_csv,
                                                                   index=False)
    use_df[(use_df['dab_year'] >= split_yrs[0]) & \
           (use_df['dab_year'] < split_yrs[1])][desired_cols]. \
               to_csv(val_csv, index=False)
    use_df[use_df['dab_year'] >= split_yrs[1]][desired_cols].to_csv(test_csv,
                                                                    index=False)


def word_tokenize(text):
    '''
    Converts text to nlp obj, tokenizes text and removes punctuation

    Input:
        text: a string of text to be tokenized

    Output:
        tokenized: a list of nlp word tokens
    '''
    tokenized = []
    # pass word list through language model.
    doc = NLP(text)
    for token in doc:
        if not token.is_punct and len(token.text.strip()) > 0:
            tokenized.append(token.text)
    return tokenized


def normalize_tokens(word_list, extra_stop=None):
    '''
    Normalizes word tokens through stemming and lemmatizing tokens,
    removing any stop words as well

    Inputs:
        word_list: list of tokenized words
        extra_stop: list of any extra stop words to be added to default
            stop word list

    Output:
        normalized: a list of normalized tokens
    '''
    normalized = []
    if isinstance(word_list, list) and len(word_list) == 1:
        word_list = word_list[0]

    if isinstance(word_list, list):
        word_list = ' '.join([str(elem) for elem in word_list])
    doc = NLP(word_list.lower())
    # add the property of stop word to words considered as stop words
    if extra_stop:
        for stopword in extra_stop:
            lexeme = NLP.vocab[stopword]
            lexeme.is_stop = True
    for word in doc:
        # if it's not a stop word or punctuation mark, add it to our article
        if word.text != '\n' and not word.is_stop and not word.is_punct \
            and not word.like_num and len(word.text.strip()) > 0:
            # we add the lematized version of the word
            normalized.append(str(word.lemma_))
    return normalized


def make_dataset(train_csv, val_csv, test_csv):
    '''
    Generates the training, validation and testing datasets as torchtext
    objects for easy incorporation with Pytorch (cleaning them in the process)

    Inputs:
        train_csv(str): name of training data csv
        val_csv(str): name of validation data csv
        test_csv(str): name of testing data csv

    Outputs:
        train: tabular dataset obj representing the training data
        test: tabular dataset obj representing the testing data
        val: tabular dataset obj representing the validation data
        text: torchtext field obj representing how text should be
            processed and stored
        label: torchtext labelfield obj representing labels should be
            processed and stored
    '''
    text = Field(sequential=True, tokenize=word_tokenize,
                 preprocessing=normalize_tokens)
    label = LabelField(dtype=torch.float)
    data_fields = [('dab_id', None), ('alj_id', None), ('alj_text', text),
                   ('decision_binary', label), ('dab_year', None)]
    train, val, test = TabularDataset.splits(path='', train=train_csv,
                                             validation=val_csv, test=test_csv,
                                             format='csv', fields=data_fields,
                                             skip_header=True)
    return train, test, val, text, label


def get_data(train_csv, val_csv, test_csv, sample_size=None):
    '''
    Grabs all data from db, cleans and splits data into test/train/validation,
    saves these sets as individual csvs (needed b/c torchtext), then reads in
    csvs, tokenizes, normalizes data and converts into TabularDataset objs
    and Field objs

    Inputs:
        train_csv(str): desired filename for training set csv
        val_csv(str): desired filename for validation set csv
        test_csv(str): desired filename for testing set csv
        sample_size(int): desired sample size for testing model
                or None for all obs to be utilized

    Outputs:
        train: tabular dataset obj representing the training data
        test: tabular dataset obj representing the testing data
        val: tabular dataset obj representing the validation data
        text: torchtext field obj representing how text should be
            processed and stored
        label: torchtext labelfield obj representing labels should be
            processed and stored
    '''
    get_split_write(train_csv, val_csv, test_csv, sample_size, (2017, 2019))
    train, test, val, text, label = make_dataset(train_csv, val_csv, test_csv)
    return train, test, val, text, label
