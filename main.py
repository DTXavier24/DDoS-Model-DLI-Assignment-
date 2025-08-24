import numpy as np
import pandas as pd

import io

import pickle # saving and loading trained model

# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# importing library for plotting
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dropout, Activation
from keras.layers import Dense # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.models import model_from_json # saving and loading trained model

from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model

from tensorflow.keras.utils import plot_model


from google.colab import drive
drive.mount('/content/drive')

Dataset1 = pd.read_csv('/content/drive/My Drive/DLI Assignment/Normal_data.csv')
Dataset2 = pd.read_csv('/content/drive/My Drive/DLI Assignment/OVS.csv')
Dataset3 = pd.read_csv('/content/drive/My Drive/DLI Assignment/metasploitable-2.csv')

df = pd.concat([Dataset1, Dataset2, Dataset3], axis=0, ignore_index=True)

print("Finisehd reading in {} entires".format(str(df.shape[0])))

metadata = ['Flow ID',
'Src IP',
'Src Port',
'Dst IP',
'Dst Port',
'Protocol',
'Timestamp',
'Flow Duration',
'Tot Fwd Pkts',
'Tot Bwd Pkts',
'TotLen Fwd Pkts',
'TotLen Bwd Pkts',
'Fwd Pkt Len Max',
'Fwd Pkt Len Min',
'Fwd Pkt Len Mean',
'Fwd Pkt Len Std',
'Bwd Pkt Len Max',
'Bwd Pkt Len Min',
'Bwd Pkt Len Mean',
'Bwd Pkt Len Std',
'Flow Byts/s',
'Flow Pkts/s',
'Flow IAT Mean',
'Flow IAT Std',
'Flow IAT Max',
'Flow IAT Min',
'Fwd IAT Tot',
'Fwd IAT Mean',
'Fwd IAT Std',
'Fwd IAT Max',
'Fwd IAT Min',
'Bwd IAT Tot',
'Bwd IAT Mean',
'Bwd IAT Std',
'Bwd IAT Max',
'Bwd IAT Min',
'Fwd PSH Flags',
'Bwd PSH Flags',
'Fwd URG Flags',
'Bwd URG Flags',
'Fwd Header Len',
'Bwd Header Len',
'Fwd Pkts/s',
'Bwd Pkts/s',
'Pkt Len Min',
'Pkt Len Max',
'Pkt Len Mean',
'Pkt Len Std',
'Pkt Len Var',
'FIN Flag Cnt',
'SYN Flag Cnt',
'RST Flag Cnt',
'PSH Flag Cnt',
'ACK Flag Cnt',
'URG Flag Cnt',
'CWE Flag Count',
'ECE Flag Cnt',
'Down/Up Ratio',
'Pkt Size Avg',
'Fwd Seg Size Avg',
'Bwd Seg Size Avg',
'Fwd Byts/b Avg',
'Fwd Pkts/b Avg',
'Fwd Blk Rate Avg',
'Bwd Byts/b Avg',
'Bwd Pkts/b Avg',
'Bwd Blk Rate Avg',
'Subflow Fwd Pkts',
'Subflow Fwd Byts',
'Subflow Bwd Pkts',
'Subflow Bwd Byts',
'Init Fwd Win Byts',
'Init Bwd Win Byts',
'Fwd Act Data Pkts',
'Fwd Seg Size Min',
'Active Mean',
'Active Std',
'Active Max',
'Active Min',
'Idle Mean',
'Idle Std',
'Idle Max',
'Idle Min',
'Label'
]
df.columns = metadata

# =============================
# Data Preprocessing and Removing NA Column (Wei Bin)
# =============================
import time
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(df):
    print("\n Preprocessing data...")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f" Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(df)

# =============================
# Model Training with Random Forest (Wei Bin)
# =============================
def train_model(X_train, y_train):
    print("Training Random Forest...")
    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=500,     # more trees → higher stability
        max_depth=60,         # deeper trees for capturing patterns
        max_features="sqrt",  # good balance for splits
        min_samples_split=2,  # allow deep branching
        min_samples_leaf=1,   # fine-grained splits
        class_weight="balanced_subsample",  # handle class imbalance better
        bootstrap=True,       # classic RF bootstrapping
        random_state=42,
        n_jobs=-1             # use all CPU cores
    )
    model.fit(X_train, y_train)

    duration = time.time() - start_time
    print(f" Training complete in {duration:.2f} seconds")
    return model

# Train model
model = train_model(X_train, y_train)
