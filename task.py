import pandas as pd
import numpy as np
import os
from scipy import signal
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt

# Define the path to the directory containing the audio files
audio_dir = 'D:/Notes/semester6/dsp-th/project/heartt/set_b'

# Define a function to load and filter an audio file
def load_and_filter_audio(filename):
    # Load the audio file using librosa
    y, sr = librosa.load(os.path.join(audio_dir, filename))

    # Apply a bandpass filter to the audio signal
    nyq = sr / 2
    low = 20
    high = 1000
    order = 4
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    sos = signal.butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    filtered_sound = signal.sosfiltfilt(sos, y)

    # Compute the Mel-frequency cepstral coefficients (MFCCs) of the filtered sound
    mfcc = librosa.feature.mfcc(y=filtered_sound, sr=sr, n_mfcc=13)

    # Get the label from the filename
    label = filename.split('_')[0]

    return pd.Series([filtered_sound, mfcc, label])


# Load the filenames of the audio files
filenames = os.listdir(audio_dir)

# Create a DataFrame to store the preprocessed data
df = pd.DataFrame({'fname': filenames})

# Apply the load_and_filter_audio function to each filename and store the results in the DataFrame
df[['filtered_sound', 'mfcc', 'label']] = df['fname'].apply(load_and_filter_audio)

# Save the preprocessed DataFrame to a CSV file
output_folder = 'D:/Notes/semester6/dsp-th/project/heartt/g'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df.to_csv(os.path.join(output_folder, 'set_k.csv'), index=False)

# Visualize the first MFCC feature of the first audio file
plt.figure(figsize=(10, 4))
ld.specshow(df['mfcc'][0], x_axis='time')
plt.colorbar()
plt.title('MFCC (dB)')
plt.tight_layout()
plt.show()
