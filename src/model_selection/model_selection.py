##################################
# LSTM
##################################
import os

import numpy as np
import requests as requests
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
from flask import Flask

from flask import jsonify


app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/data/model', methods=['POST'])
def run_model_selection():
    # pick a large window size of 50 cycles
    sequence_length = 50

    # function to reshape features into (samples, time steps, features)
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


    # pick the feature columns
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)

    def get_training_data_as_dataframe():
        response = requests.get(os.environ['TRAININGDB_PREPROCESSED'])
        training_data = response.json()
        return pd.DataFrame.from_dict(training_data)

    train_df = get_training_data_as_dataframe()

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
    label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['label1'])
                 for id in train_df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)

    # Next, we build a deep network.
    # The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units.
    # Dropout is also applied after each LSTM layer to control overfitting.
    # Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem.
    # build the network
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]

    model = Sequential()

    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=nb_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Saving model in a given location provided as an env. variable
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model.h5")
        model.save(file_path)
        return jsonify({'message': f"Saved the model to the location : {model_repo}"}), 200
    else:
        model.save("model.h5")
        return jsonify({'message': 'The model was saved locally.'}), 200


app.run(host='0.0.0.0', port=5000)
