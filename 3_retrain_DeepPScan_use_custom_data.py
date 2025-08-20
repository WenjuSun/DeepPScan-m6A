#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : 3_retrain_DeepPScan_use_custom_data.py
@Time    : 2024/08/15 11:00:32
@Author  : Wenju Sun
@Version : 1.0
@Contact : wenju.sun@qq.com
@License : Copyright (C) - All Rights Reserved Sun Wenju
@Desc    : None
@Usage   : None
'''

import argparse
import logging
import random
random.seed(1234)
# 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 
import tensorflow as tf
import keras.layers as kl
from keras.models import Sequential, Model
from keras.layers import Input, InputLayer, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def create_custom_logger(name):
    formatter = logging.Formatter(fmt='[%(asctime)s][%(levelname)s][%(module)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

logger = create_custom_logger('root')


# custom functions defination
# Constants
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
# Define a function for encoding amino acid sequences using one-hot encoding.
def one_hot_encode(sequence, encode_len):
    encoded = to_categorical([amino_acids.index(aa) for aa in sequence], num_classes=len(amino_acids), dtype="int")
    if len(encoded) < encode_len:
        encoded = np.pad(encoded, ((0, encode_len - len(encoded)), (0, 0)), 'constant')
    return encoded


def preprocess(file_path, window_len, step_size):
    # Each line of the input file must consist of three columns: <protein_id>\tab<protein_sequence>\tab<preference_score>
    df = pd.read_csv(file_path, sep="\t", header=0)
    new_protein_ids = []
    new_sequences = []
    new_scores = []
    for index, row in df.iterrows():
        protein_id = row['protein_id']
        sequence = row['protein_sequence']
        score = row['preference_score']
        if len(sequence) > window_len:
            for i in range(0, len(sequence)-window_len+1, step_size):
                subseq = sequence[i:i+window_len]
                new_protein_ids.append(protein_id)
                new_sequences.append(subseq)
                new_scores.append(score)
            # The construction method of the last sequence does not involve padding with zeros.
            if (len(sequence)-window_len) % step_size != 0:
                subseq = sequence[len(sequence)-window_len:]
                new_protein_ids.append(protein_id)
                new_sequences.append(subseq)
                new_scores.append(score)
        else:
            new_protein_ids.append(protein_id)
            new_sequences.append(sequence)
            new_scores.append(score)
    #
    encoded_sequences = [one_hot_encode(aa_seq, window_len) for aa_seq in new_sequences]
    processed_df = pd.DataFrame({
        'protein_id': new_protein_ids,
        'encoded_AAstr': encoded_sequences,
        'score': new_scores,
    })
    return processed_df


def tf_Spearman(y_true, y_pred):
    return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32) )


def DeepPScan(model_params, encoded_AAstr_len):
    input = Input(shape=(encoded_AAstr_len, 20))
    x = Conv1D(model_params['num_filters1'], kernel_size=model_params['kernel_size1'], padding=model_params['pad'], name='Conv1D_1')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3)(x)
    for i in range(1, model_params['n_conv_layer']):
        x = Conv1D(model_params['num_filters'+str(i+1)],
                        kernel_size=model_params['kernel_size'+str(i+1)],
                        padding=model_params['pad'],
                        name=str('Conv1D_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
    #
    x = Flatten()(x)
    # dense layers
    for i in range(0, model_params['n_add_layer']):
        x = Dense(model_params['dense_neurons'+str(i+1)], name=str('Dense_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(model_params['dropout_prob'])(x)
    bottleneck = x
    # 
    outputs = Dense(1, activation='linear', name=str('m6A_log2OR'))(bottleneck)
    model = Model(input, outputs)
    model.compile(Adam(learning_rate=model_params['lr']), loss='mse', metrics=[tf_Spearman])
    return model


def train(selected_model, X_train, Y_train, X_valid, Y_valid, params):
    with tf.device('/GPU:0'):
        mt_history=selected_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                                        batch_size=params['batch_size'], epochs=params['epochs'],
                                        callbacks=[EarlyStopping(patience=params['early_stop'], monitor="val_loss", restore_best_weights=True),
                                        History()])
        return selected_model, mt_history


def summary_statistics(model, batch_size, X, Y, set_label):
    pred = model.predict(X, batch_size=batch_size)
    print(set_label + ' RMSE ' + ' = ' + str("{0:0.2f}".format(mean_squared_error(Y, pred.squeeze()))))
    print(set_label + ' PCC ' + ' = ' + str("{0:0.2f}".format(pearsonr(Y, pred.squeeze())[0])))
    print(set_label + ' SCC ' + ' = ' + str("{0:0.2f}".format(spearmanr(Y, pred.squeeze())[0])))


def tph_visualization(mt_history, pdf_file):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.92, wspace=0.14)
    fig.suptitle("Visualization of training history")
    # ax1
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(mt_history.history['loss'], label='Train Loss')
    ax1.plot(mt_history.history['val_loss'], label='Validation Loss')
    min_val_loss = min(mt_history.history['val_loss'])
    ax1.axvline(x=mt_history.history['val_loss'].index(min_val_loss), color='red', linestyle='--')
    ax1.set_title('Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    # ax2
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(mt_history.history['tf_Spearman'], label='Train Spearman')
    ax2.plot(mt_history.history['val_tf_Spearman'], label='Validation Spearman')
    max_val_tf_Spearman = max(mt_history.history['val_tf_Spearman'])
    ax2.axvline(x=mt_history.history['val_tf_Spearman'].index(max_val_tf_Spearman), color='red', linestyle='--')
    ax2.set_title('Spearman Correlation Coefficient Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Spearman Correlation Coefficient')
    ax2.legend()
    fig.savefig(pdf_file)


def my_parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-i', '--input', type=str, required=True, help=r'The sequences and scores input file is used for model training. Each line consists of three columns, with the format being <protein_id>\tab<protein_sequence>\tab<preference_score>')
    my_parser.add_argument('-l', '--length', type=int, default=300, help='The length of the peptides used for training, default: 300.')
    my_parser.add_argument('-s', '--step', type=int, default=30, help='The step size for sliding window on the protein sequence, default: 30.')
    my_parser.add_argument('-o', '--output', type=str, required=True, help='The prefix name of output model files.')
    my_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    return my_parser.parse_args()


def main():
    my_args = my_parse_args()
    input_file = my_args.input
    window_len = my_args.length
    step_size = my_args.step
    output_prefix = my_args.output

    logger.info(f"command line: python3 3_retrain_DeepPScan_use_custom_data.py -i {input_file} -l {window_len} -s {step_size} -o {output_prefix}")

    # coding begin here
    input_preprocessed_df = preprocess(input_file, window_len=window_len, step_size=step_size)
    # training dataset: 70%,  test dataset: 30%
    training_df, validation_df = train_test_split(input_preprocessed_df, test_size=0.3, random_state=123)
    logger.info(f"training_df_shape:{training_df.shape}...")
    logger.info(f"validation_df_shape:{validation_df.shape}...")
    # extract features and labels
    X_train =  np.array(training_df['encoded_AAstr'].tolist())
    y_train = training_df['score'].values
    X_val =  np.array(validation_df['encoded_AAstr'].tolist())
    y_val = validation_df['score'].values
    logger.info(f"X_train_shape:{X_train.shape}; y_train_shape:{y_train.shape}...")
    logger.info(f"X_val_shape:{X_val.shape}; y_val_shape:{y_val.shape}...")

    # build model
    params_t1 = {'lr': 0.002,
            'n_conv_layer': 5,
            'num_filters1': 512,
            'num_filters2': 256,
            'num_filters3': 128,
            'num_filters4': 128,
            'num_filters5': 256,
            'kernel_size1': 15,
            'kernel_size2': 9,
            'kernel_size3': 7,
            'kernel_size4': 5,
            'kernel_size5': 3,
            'n_add_layer': 3,
            'dense_neurons1': 512,
            'dense_neurons2': 256,
            'dense_neurons3': 256,
            'dropout_prob': 0.4,
            'pad': 'same'}
    model_t1 = DeepPScan(params_t1, window_len)
    # model_t1.summary()

    # training the CNN model
    train_params = {'batch_size': 128, 'epochs': 500, 'early_stop': 50}
    model_t1, mt_history = train(model_t1, X_train, y_train, X_val, y_val, train_params)
    # 
    summary_statistics(model_t1, train_params["batch_size"], X_train, y_train, "train")
    summary_statistics(model_t1, train_params["batch_size"], X_val, y_val, "validation")
    #
    pdf_file = output_prefix + ".train_history_visualization.pdf"
    tph_visualization(mt_history, pdf_file)
    # save model and weights
    model_json = model_t1.to_json()
    with open(output_prefix + '.model.json', "w") as json_file:
        json_file.write(model_json)
    model_t1.save_weights(output_prefix + '.model.h5')


    logger.info('Run completed!')


if __name__ == '__main__':
    main()

