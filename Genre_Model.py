from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


enc = LabelBinarizer()
Labels_enc = enc.fit_transform(Labels.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(Data, Labels_enc,test_size=0.15 , random_state=42 , shuffle=True)

from keras.models import Sequential
from keras.layers import Dense,AveragePooling2D, MaxPooling2D, Conv2D, Flatten, Dropout
# Model Definition
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