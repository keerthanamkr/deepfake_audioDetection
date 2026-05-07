import soundfile
import librosa
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
import pickle
from keras.models import Sequential
from sklearn.metrics import accuracy_score

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, LSTM
from keras.layers import Conv2D
from keras.models import Sequential, load_model, Model

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    sound_file.close()        
    return result

path = "Dataset"
'''
X = []
Y = []
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        mfcc = extract_feature(root+"/"+directory[j], mfcc=True, chroma=True, mel=True)
        label = 0
        if name == 'Original':
            label = 1
        mfcc = np.reshape(mfcc, (10, 6, 3))    
        X.append([mfcc])
        Y.append(label)
        print(str(mfcc.shape)+" "+str(label)+" "+name)

X = np.asarray(X)
Y = np.asarray(Y)

np.save("model/X", X)
np.save("model/Y", Y)
'''

X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
print(X.shape)
X = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3] * X.shape[4])))
print(X.shape)               
scaler = MinMaxScaler((0,1))
X = scaler.fit_transform(X)
X = np.reshape(X, (X.shape[0], 1, 10, 6, 3))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

lstm_model = Sequential()
lstm_model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (X.shape[1], X.shape[2], X.shape[3], X.shape[4])))
lstm_model.add(TimeDistributed(MaxPooling2D((4, 4))))
lstm_model.add(Dropout(0.5))
lstm_model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
lstm_model.add(TimeDistributed(MaxPooling2D((1, 1))))
lstm_model.add(Dropout(0.5))
lstm_model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
lstm_model.add(TimeDistributed(MaxPooling2D((1, 1))))
lstm_model.add(Dropout(0.5))
lstm_model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
lstm_model.add(TimeDistributed(MaxPooling2D((1, 1))))
lstm_model.add(Dropout(0.5))
lstm_model.add(TimeDistributed(Flatten()))
lstm_model.add(LSTM(32))
lstm_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_lstm.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_lstm.hdf5', verbose = 1, save_best_only = True)
    hist = lstm_model.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_lstm_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    lstm_model.load_weights("model/cnn_lstm.hdf5")
   
predict = lstm_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)





