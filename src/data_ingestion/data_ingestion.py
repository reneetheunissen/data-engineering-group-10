##################################
# Data Ingestion
# This code belongs to Umberto Griffo and is adjusted and containerized by Group 10.
# Original code: https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM
##################################
import json

import pandas as pd
from flask import Flask, Response
from pandas import DataFrame

from resources.db_util import DBUtil

app = Flask(__name__)
app.config["DEBUG"] = True
db_util = DBUtil()


@app.route('/data/<table_name>', methods=['POST'])
def create_table(table_name: str):
    # Select the correct data
    if table_name == 'train':
        df = pd.read_csv('Dataset/PM_train.txt', sep=" ", header=None)
        df = prepare_train_or_test_df(df)
    elif table_name == 'test':
        df = pd.read_csv('Dataset/PM_test.txt', sep=" ", header=None)
        df = prepare_train_or_test_df(df)
    elif table_name == 'truth':
        df = pd.read_csv('Dataset/PM_truth.txt', sep=" ", header=None)
        df.drop(df.columns[1], axis=1, inplace=True)
        df.columns = ['value']
    else:
        return
    # Create the table
    db_util.create_tb(table_name=table_name, column_names=df.columns)
    json_df = df.to_json(orient='records')
    json_df = json.loads(json_df)
    db_util.add_data_records(table_name=table_name, records=json_df)
    # Report success
    return json.dumps({'message': f'the {table_name} table was created at /data/{table_name}'}, indent=4), 200


def prepare_train_or_test_df(df: DataFrame) -> DataFrame:
    df.drop(df.columns[[26, 27]], axis=1, inplace=True)
    df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                  's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                  's15', 's16', 's17', 's18', 's19', 's20', 's21']
    return df.sort_values(['id', 'cycle'])


@app.route('/data/<table_name>', methods=['GET'])
def read_data(table_name):
    df = db_util.read_data_records(table_name)
    resp = Response(df.to_json(orient='records'), status=200, mimetype='application/json')
    return resp


app.run(host='0.0.0.0', port=7270)
