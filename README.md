# DeepPScan-m6A
DeepPScan-m6A is a deep learning model based on convolutional neural networks. It can quantitatively predict the m6A preference score of protein sequences, which is used to discover potential m6A reader proteins or polypeptides. If the user has preference scores for other modifications or constructs, and uses custom new data to retrain the model, then the model can be expanded to predict various preference scores of proteins.

# System Requirements

## Hardware requirements

All the scripts in this package have been tested on a Dell 940xa server equipped with NVIDIA Tesla V100 GPUs. Depending on the volume of data, the machine requires sufficient memory and GPU memory.

## Software requirements

### OS Requirements

All the scripts and codes in this package have been tested on the CentOS 7.9 Linux system.

### Python Dependencies

```bash
logging
argparse
random
numpy
pandas
scipy
sklearn
tensorflow
keras

```

## Environment Deployment

```bash
mamba create --name DeepPScan python=3.7 tensorflow-gpu=1.14.0 keras-gpu=2.2.4
mamba activate DeepPScan
mamba install ipykernel=5.4.3 numpy=1.16.2 pandas=0.25.3 scipy=1.7.3 scikit-learn=1.0.2 matplotlib=3.1.1 

```

# Installation and Usage

```bash
# Install from Github
git clone https://github.com/WenjuSun/DeepPScan-m6A.git
cd DeepPScan-m6A
mamba activate DeepPScan

```

## Run the prediction for the new sequence using the model in the command line.

```bash
# Use the saved model and weights to predict the new protein sequence
python 4_prediction_for_custom_sequences.py -m RBP_m6A_preference_CNN_model_DeepPScan -i 4_merged_data_for_predict_test.txt -o 4_merged_data_for_predict_test

# You can learn about the corresponding parameters and their functions by referring to the "help" information.
python 4_prediction_for_custom_sequences.py -h
# usage: 4_prediction_for_custom_sequences.py [-h] -m MODEL -i INPUT [-l LENGTH]
#                                             [-s STEP] -o OUTPUT [-v]
# optional arguments:
#   -h, --help            show this help message and exit
#   -m MODEL, --model MODEL
#                         The prefix name of model files.
#   -i INPUT, --input INPUT
#                         The sequence input file for prediction. Each line
#                         consists of two columns, with the format being
#                         <protein_id>\tab<protein_sequence>
#   -l LENGTH, --length LENGTH
#                         The length of the peptides used for predicted,
#                         default: 300. It needs to be consistent with the
#                         length used during model training.
#   -s STEP, --step STEP  The step size for sliding window on the protein
#                         sequence, default: 30.
#   -o OUTPUT, --output OUTPUT
#                         The prefix name of output files.
#   -v, --verbose         Enable verbose mode

```

## Re-train the model using custom data in the command line

```bash
# Or use custom data to rebuild the model
python 3_retrain_DeepPScan_use_custom_data.py -i 0_your_data_for_retrain.txt -o retrain_your_CNN_model_DeepPScan

# You can learn about the corresponding parameters and their functions by referring to the "help" information.
python 3_retrain_DeepPScan_use_custom_data.py -h
# usage: 3_retrain_DeepPScan_use_custom_data.py [-h] -i INPUT [-l LENGTH]
#                                               [-s STEP] -o OUTPUT [-v]
# optional arguments:
#   -h, --help            show this help message and exit
#   -i INPUT, --input INPUT
#                         The sequences and scores input file is used for model
#                         training. Each line consists of three columns, with
#                         the format being <protein_id>\tab<protein_sequence>
#                         ab<preference_score>
#   -l LENGTH, --length LENGTH
#                         The length of the peptides used for training, default:
#                         300.
#   -s STEP, --step STEP  The step size for sliding window on the protein
#                         sequence, default: 30.
#   -o OUTPUT, --output OUTPUT
#                         The prefix name of output model files.
#   -v, --verbose         Enable verbose mode

```

## Examples and Tutorial Notebook

- You can open the notebook:  [0_train_CNN_model_for_RBP_m6A_log2OR_predicting.ipynb](0_train_CNN_model_for_RBP_m6A_log2OR_predicting.ipynb)

  > Refer to the practice code in it and test and run it step by step within the notebook to understand the detailed model training process. 

- Or open the notebook: [1_predict_RBP_m6A_log2OR_for_AAseq.ipynb](1_predict_RBP_m6A_log2OR_for_AAseq.ipynb)

  > Refer to the practice code in it, and test and run it step by step within the notebook to understand the detailed process of using the model to predict new sequences.



# Citing `DeepPScan-m6A`

> #TODO
>
> doi: xxxx



# License

This project is covered under the **MIT License**.
