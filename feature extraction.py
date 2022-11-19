
# Feature Extraction (mel spectrogram)
import numpy as np
import librosa
import os
import glob

X = []
y = []
dir=" "
for i, audio_path in enumerate(glob.glob(dir + "*.wav")):
    filename = os.path.basename(audio_path)
    y, sr = librosa.load(audio_path, sr=22050)
    F0=librosa.feature.melspectrogram(y ,sr=22050 , n_fft=2048 , hop_length=512)
    #F1=librosa.feature.chroma_stft(y ,sr=22050 , n_fft=2048 , hop_length=512)
    #F2=librosa.feature.spectral_contrast(y ,sr=22050 , n_fft=2048 , hop_length=512)
    # np.save(os.path.join(dir, filename + ".npy"), F0)
    X.append(F0)
    y.append(str(filename[0:3]))

Data = np.array(X)
#np.save("Data.npy",Data)
Labels = np.array (y)
#np.save("Labels.npy",Labels)
