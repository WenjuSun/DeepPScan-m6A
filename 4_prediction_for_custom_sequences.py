#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : 4_prediction_for_custom_sequences.py
@Time    : 2024/08/19 10:37:43
@Author  : Wenju Sun
@Version : 1.0
@Contact : wenju.sun@qq.com
@License : Copyright (C) - All Rights Reserved Sun Wenju
@Desc    : None
@Usage   : None
'''

import argparse
import logging
# 
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras.utils import to_categorical


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
def GPU_settings():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def load_CNN_model(model_prefix):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    keras_model_json = model_prefix + '.json'
    keras_model_weights = model_prefix + '.h5'
    my_model = model_from_json(open(keras_model_json).read())
    my_model.load_weights(keras_model_weights)
    return my_model


# Define a function for encoding amino acid sequences using one-hot encoding.
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
def one_hot_encode(sequence, encode_len):
    encoded = to_categorical([amino_acids.index(aa) for aa in sequence], num_classes=len(amino_acids), dtype="int")
    if len(encoded) < encode_len:
        encoded = np.pad(encoded, ((0, encode_len - len(encoded)), (0, 0)), 'constant')
    return encoded


def preprocess_input(file_path, window_len, step_size):
    input_df = pd.read_csv(file_path, sep="\t", header=None)
    protein_id_list = []
    protein_AAstr_list = []
    peptide_id_list = []
    peptide_AAstr_list = []
    new_AAseqStr_list = []
    for index, row in input_df.iterrows():
        protein_id = row[0]
        sequence = row[1]
        if len(sequence) > window_len:
            for i in range(0, len(sequence)-window_len+1, step_size):
                peptide_AAstr = sequence[i:i+window_len]
                peptide_id = str(protein_id) + ':' + str(i+1)+ '-' + str(i+window_len+1)
                protein_id_list.append(protein_id)
                protein_AAstr_list.append(sequence)
                peptide_id_list.append(peptide_id)
                peptide_AAstr_list.append(peptide_AAstr)
                new_AAseqStr_list.append(peptide_AAstr)
            # The construction method of the last sequence does not involve padding with zeros.
            if (len(sequence)-window_len) % step_size != 0:
                peptide_AAstr = sequence[len(sequence)-window_len:]
                peptide_id = str(protein_id) + ':' + str(len(sequence)-window_len+1)+ '-' + str(len(sequence))
                protein_id_list.append(protein_id)
                protein_AAstr_list.append(sequence)
                peptide_id_list.append(peptide_id)
                peptide_AAstr_list.append(peptide_AAstr)
                new_AAseqStr_list.append(peptide_AAstr)
        else:
            protein_id_list.append(protein_id)
            protein_AAstr_list.append(sequence)
            peptide_id_list.append(protein_id)
            peptide_AAstr_list.append(sequence)
            new_AAseqStr_list.append(sequence)
    # one_hot_encode
    encoded_sequences = [one_hot_encode(aa_seq, window_len) for aa_seq in new_AAseqStr_list]
    processed_df = pd.DataFrame({
        'protein_id': protein_id_list,
        'protein_AAstr': protein_AAstr_list,
        'peptide_id': peptide_id_list,
        'peptide_AAstr': peptide_AAstr_list,
        'encoded_AAstr': encoded_sequences,
    })
    return processed_df



def my_parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-m', '--model', type=str, required=True, help='The prefix name of model files.')
    my_parser.add_argument('-i', '--input', type=str, required=True, help=r'The sequence input file for prediction. Each line consists of two columns, with the format being <protein_id>\tab<protein_sequence>')
    my_parser.add_argument('-l', '--length', type=int, default=300, help='The length of the peptides used for predicted, default: 300. It needs to be consistent with the length used during model training.')
    my_parser.add_argument('-s', '--step', type=int, default=30, help='The step size for sliding window on the protein sequence, default: 30.')
    my_parser.add_argument('-o', '--output', type=str, required=True, help='The prefix name of output files.')
    my_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    return my_parser.parse_args()


def main():
    my_args = my_parse_args()
    model_prefix = my_args.model
    input_seq_file = my_args.input
    window_len = my_args.length
    step_size = my_args.step
    output_prefix = my_args.output

    logger.info(f"command line: python3 4_prediction_for_custom_sequences.py -m {model_prefix} -i {input_seq_file} -l {window_len} -s {step_size} -o {output_prefix}")

    # Configure the GPU and load the model
    GPU_settings()
    my_model = load_CNN_model(model_prefix)
    # my_model.summary()
    processed_df = preprocess_input(input_seq_file, window_len, step_size)
    logger.info(f"Shape of the processed input: {processed_df.shape}")
    X =  np.array(processed_df['encoded_AAstr'].tolist())
    logger.info(f"Shape of the one_hot_encoded input: {X.shape}")
    # prediction
    with tf.device('/GPU:0'):
        logger.info(f"predicting......")
        predict_result = my_model.predict(X)
        logger.info(f"Shape of the prediction result: {predict_result.shape}")
    # Construct the result and output.
    processed_df["predict_score"] = predict_result.flatten().tolist()
    processed_df = processed_df.drop("encoded_AAstr", axis=1)

    # peptide_level_output
    peptide_level_output = output_prefix + ".peptide_wtih_score.xls"
    processed_df.to_csv(peptide_level_output, sep="\t", header=True, index=False)
    logger.info(f"Shape of all prediction result: {processed_df.shape}")
    
    # protein_level_output
    # For each protein, take the maximum log2OR value from the prediction results of all peptide segments.
    protein_level_output = output_prefix + ".protein_max_score.xls"
    idx = processed_df.groupby('protein_id')['predict_score'].idxmax()
    protein_result_df = processed_df.loc[idx].reset_index(drop=True)
    protein_result_df.to_csv(protein_level_output, sep="\t", header=True, index=False)
    logger.info(f"Shape of the protein max score result: {protein_result_df.shape}")


    logger.info('Run completed!')


if __name__ == '__main__':
    main()

