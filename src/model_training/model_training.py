import os

import keras
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify
from keras.models import load_model

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/data/model', methods=['POST'])
def train_model():
    model_repo = os.environ['MODEL_REPO']
    model_path = os.path.join(model_repo, "model.h5")
    model = load_model(model_path)

    def get_training_data_as_dataframe():
        response = requests.get(os.environ['TRAININGDB_PREPROCESSED'])
        training_data = response.json()
        return pd.DataFrame.from_dict(training_data)

    train_df = get_training_data_as_dataframe()

    def gen_sequence(id_df, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows and sequence_length is equal to 50
        # so zip iterate over two following list of numbers (0,112),(50,192)
        # 0 50 -> from row 0 to row 50
        # 1 51 -> from row 1 to row 51
        # 2 52 -> from row 2 to row 52
        # ...
        # 111 191 -> from row 111 to 191
        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    # generator for the sequences
    seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
               for id in train_df['id'].unique())

    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

    # function to generate labels
    def gen_labels(id_df, seq_length, label):
        # For one id I put all the labels in a single matrix.
        # For example:
        # [[1]
        # [4]
        # [1]
        # [5]
        # [9]
        # ...
        # [200]]
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        # I have to remove the first seq_length labels
        # because for one id the first sequence of seq_length size have as target
        # the last label (the previus ones are discarded).
        # All the next id's sequences will have associated step by step one label as target.
        return data_matrix[seq_length:num_elements, :]

    # generate labels
    label_gen = [gen_labels(train_df[train_df['id'] == id], 50, ['label1'])
                 for id in train_df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)

    # fit the network
    model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                         mode='min'),
                           keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,
                                                           mode='min', verbose=0)]
              )

    # Saving model in a given location provided as an env. variable
    model.save(model_path)
    return jsonify({'message': f"Saved the trained model to the location : {model_repo}"}), 200


app.run(host='0.0.0.0', port=5000)