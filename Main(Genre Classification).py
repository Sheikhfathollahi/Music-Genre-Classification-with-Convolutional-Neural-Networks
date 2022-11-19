# Import necessary libraries
from pydub import AudioSegment
import speech_recognition as sr
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense,AveragePooling2D, MaxPooling2D, Conv2D, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import itertools


# chunk datset into 3-seconds pieces with a 50% overlap

inputdir = ''
outputdir = ''
# Input audio file to be sliced
for filename in os.listdir(inputdir):
    save_file_name = filename[:-4]
    audio = AudioSegment.from_file(inputdir + "/" + filename, "wav")
    n = len(audio)
    counter = 1
    interval = 3 * 1000
    overlap = 1.5 * 1000
    start = 0
    end = 0
    flag = 0
    for i in range(0, 2 * n, interval):
        if i == 0:
            start = 0
            end = interval
        else:
            start = end - overlap
            end = start + interval
        if end >= n:
            end = n
            flag = 1
            # Storing audio file from the defined start to end
        chunk = audio[start:end]
        # Filename / Path to store the sliced audio
        chunk_name = save_file_name + "_{0}.wav".format(i)

        # Store the sliced audio file to the defined path
        chunk.export(outputdir + "/" + chunk_name, format="wav")
        # Print information about the current chunk
        print("Processing chunk " + str(counter) + ". Start = "
              + str(start) + " end = " + str(end))

# Feature Extraction (mel spectrogram)
X = []
y = []
for i, audio_path in enumerate(glob.glob(outputdir + "*.wav")):
    filename = os.path.basename(audio_path)
    y, sr = librosa.load(audio_path, sr=22050)
    F0=librosa.feature.melspectrogram(y ,sr=22050 , n_fft=2048 , hop_length=512)
    #F1=librosa.feature.chroma_stft(y ,sr=22050 , n_fft=2048 , hop_length=512)
    #F2=librosa.feature.spectral_contrast(y ,sr=22050 , n_fft=2048 , hop_length=512)
    # np.save(os.path.join(dir, filename + ".npy"), F0)
    X.append(F0)
    y.append(str(filename[0:3]))

Data = np.array(X)
Labels = np.array (y)

enc = LabelBinarizer()
Labels_enc = enc.fit_transform(Labels.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(Data, Labels_enc,test_size=0.15 , random_state=42 , shuffle=True)

# Model
input_shape = ''
num_genres = 10
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ) )
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.4))

# MLP
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='selu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
#from keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=1, verbose=0, mode='auto')
history_cnn = model.fit(X_train, y_train,
                      epochs=50,
                      batch_size=64,
                      verbose=2,
                      validation_data=(X_test, y_test))


#Validation

plt.plot(history_cnn.history['accuracy'], c="r", label="CNN - train set")
plt.plot(history_cnn.history["val_accuracy"], c="b", linestyle="--", label="CNN - test set")
plt.legend()
plt.title("Evolution of the learning of both models")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim(0, 40)
plt.ylim(0, 1)
plt.show()

#plt.figure(figsize=(20,12))
plt.plot(history_cnn.history["loss"], c="r", label="CNN - train set")
plt.plot(history_cnn.history["val_loss"], c="b", linestyle="--", label="CNN - test set")
plt.title("Evolution of the learning of both models")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xlim(0, 40)
plt.ylim(0, 1)
plt.show()


#Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

X_test=X_test.reshape('','','','')
y_pred_2 = model.predict(X_test)
y_enc = enc.fit_transform(y_test.reshape(-1, 1))
y_pred_2 = enc.inverse_transform(y_pred_2)

cnf_matrix_2 = confusion_matrix(y_pred_2, enc.inverse_transform(y_test), labels=enc.classes_ )
#plt.figure(figsize=(20,20))
plot_confusion_matrix(cnf_matrix_2, classes=enc.classes_, title='Confusion matrix, using CNN')
plt.show()

