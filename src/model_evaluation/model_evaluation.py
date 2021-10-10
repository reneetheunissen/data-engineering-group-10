##################################
# EVALUATE ON TEST DATA
# This code belongs to Umberto Griffo and is containerized by Group 10.
# Original code: https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM
##################################
import os

import numpy as np
import pandas as pd
import requests
from flask import jsonify, Flask
from keras.models import load_model
from sklearn.metrics import recall_score, precision_score

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/evaluation/results', methods=['POST'])
def run_model_evaluation():
    # We pick the last sequence for each id in the test data
    def get_test_data_as_dataframe():
        response = requests.get(os.environ['TESTDB_PREPROCESSED'])
        training_data = response.json()
        return pd.DataFrame.from_dict(training_data)

    test_df = get_test_data_as_dataframe()

    # pick the feature columns
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)

    seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-50:]
                           for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= 50]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    # Similarly, we pick the labels
    y_mask = [len(test_df[test_df['id']==id]) >= 50 for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(int)
    label_array_test_last = [x[0] for x in label_array_test_last]
    # if best iteration's model was saved then load and use it
    model_repo = os.environ['MODEL_REPO']
    model_path = os.path.join(model_repo, "model.h5")
    estimator = load_model(model_path)

    # make predictions and compute confusion matrix
    y_pred_test = (estimator.predict(seq_array_test_last) > 0.5).astype("int32")
    y_true_test = label_array_test_last

    # compute precision and recall
    precision_test = precision_score(y_true_test, y_pred_test)
    recall_test = recall_score(y_true_test, y_pred_test)
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
    return jsonify({'message': f'Precision: {precision_test}, Recall: {recall_test}, F1-score: {f1_test}'}), 200


app.run(host='0.0.0.0', port=7274)
