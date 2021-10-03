##################################
# Data Preprocessing
##################################
import json
import os

import numpy as np
import pandas as pd
import requests
from flask import Flask
from sklearn import preprocessing

from resources.db_util import DBUtil

app = Flask(__name__)
app.config["DEBUG"] = True
db_util = DBUtil()


@app.route('/data/', methods=['POST'])
def run_data_preprocessing():
    #######
    # TRAIN
    #######
    def get_training_data_as_dataframe():
        response = requests.get(os.environ['TRAININGDB_API'])
        training_data = response.json()
        return pd.DataFrame.from_dict(training_data)

    train_df = get_training_data_as_dataframe()
    # Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)
    # generate label columns for training data
    # we will only make use of "label1" for binary classification,
    # while trying to answer the question: is a specific engine going to fail within w1 cycles?
    w1 = 30
    w0 = 15
    train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
    train_df['label2'] = train_df['label1']
    train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

    # MinMax normalization (from 0 to 1)
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)

    db_util.create_tb('train_preprocessed', train_df.columns)

    ######
    # TEST
    ######
    def get_test_data_as_dataframe():
        response = requests.get(os.environ['TESTDB_API'])
        test_data = response.json()
        return pd.DataFrame.from_dict(test_data)

    test_df = get_test_data_as_dataframe()
    # MinMax normalization (from 0 to 1)
    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize,
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)


    # We use the ground truth dataset to generate labels for the test data.
    # generate column max for test data
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']

    def get_truth_data_as_dataframe():
        response = requests.get(os.environ['TRUTHDB_API'])
        truth_data = response.json()
        return pd.DataFrame.from_dict(truth_data)

    truth_df = get_truth_data_as_dataframe()

    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)

    # generate RUL for test data
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)

    # generate label columns w0 and w1 for test data
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
    test_df['label2'] = test_df['label1']
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
    db_util.create_tb('test_preprocessed', test_df.columns)
    db_util.create_tb('truth_preprocessed', truth_df.columns)
    return json.dumps({'message': 'all data was preprocessed and updated'}, sort_keys=False, indent=4), 200
