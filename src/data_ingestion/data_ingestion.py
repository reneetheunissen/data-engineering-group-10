##################################
# Data Ingestion
# This code belongs to Umberto Griffo and is containerized by Group 10.
# Original code: https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM
##################################
import json

import pandas as pd
from flask import Flask
from resources.db_util import DBUtil

app = Flask(__name__)
app.config["DEBUG"] = True
db_util = DBUtil()


@app.route('/data/train', methods=['POST'])
def _read_training_data():
    # read training data - It is the aircraft engine run-to-failure data.
    train_df = pd.read_csv('../../Dataset/PM_train.txt', sep=" ", header=None)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_df = train_df.sort_values(['id','cycle'])
    db_util.create_tb('train', train_df.columns)
    return json.dumps({'message': 'the training table was created'}, sort_keys=False, indent=4), 200


@app.route('/data/test', methods=['POST'])
def _read_test_data():
    # read test data - It is the aircraft engine operating data without failure events recorded.
    test_df = pd.read_csv('../../Dataset/PM_test.txt', sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    db_util.create_tb('test', test_df.columns)
    return json.dumps({'message': 'the test table was created'}, sort_keys=False, indent=4), 200


@app.route('/data/truth', methods=['POST'])
def _read_ground_truth_data():
    # read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
    truth_df = pd.read_csv('../../Dataset/PM_truth.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    db_util.create_tb('truth', truth_df.columns)
    return json.dumps({'message': 'the ground truth table was created'}, sort_keys=False, indent=4), 200


app.run(host='0.0.0.0', port=5000)
