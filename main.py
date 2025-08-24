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

from scipy.stats import zscore

def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(",".join(result))

def analyze(df):
    print()
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

import matplotlib.pyplot as plt
from matplotlib.pyplot import *

plt.figure(figsize=(20,20))

fig, ax = plt.subplots(figsize=(20,20))

class_distribution = df['Label'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')

# Before Cleaning Data set for Duplicate
sorted_ds = np.argsort(-class_distribution.values)
for i in sorted_ds:
    print('Number of data points in class', class_distribution.index[i],':', class_distribution.values[i],
          '(', np.round((class_distribution.values[i]/df.shape[0]*100), 3), '%)')
plt.ylabel('Data points per Class')
plt.title('Distribution of InSDN Training Data Before Cleaning')
plt.grid()
plt.show()
