import librosa
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# Set the path to your data folder
data_path = 'D:/Notes/semester6/dsp-th/project/heartt/set_b'

# Load dataset
X, y = [], []
max_len = 0
for file_name in os.listdir(data_path):
    if file_name.endswith('.wav'):
        file_path = os.path.join(data_path, file_name)
        if file_name.startswith('bunlabelled'):
            label = 0
        elif file_name.startswith('murmur'):
            label = 1
        elif file_name.startswith('normal'):
            label = 2
        elif file_name.startswith('extrastole'):
            label = 3
        else:
            continue
        y.append(label)
        y_wave, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y_wave, sr=sr, n_mfcc=13)
        X.append(mfcc.T)
        if len(mfcc.T) > max_len:
            max_len = len(mfcc.T)
print('Number of samples:', len(X), len(y))
X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# Check if there are samples of both classes in the dataset
if len(set(y)) < 2:
    print('Error: Dataset contains samples of only one class')
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Number of train samples:', len(X_train), len(y_train))
    print('Number of test samples:', len(X_test), len(y_test))
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Accuracy:', score)
