import numpy as np
import math
import matplotlib.pyplot as plt
from pydub import AudioSegment
import soundfile as sf
from pydub.utils import make_chunks
import glob
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import librosa
import librosa.display
from keras import models
from keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from keras.models import load_model
import os, re, os.path
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score

def partitionnate(folder, duration):
    """Permet de partitionner les signaux test avec une durée duration dans le dossier folder"""
    for instru in os.listdir(folder):
        instru_folder = folder + "/" + instru
        for j in os.listdir(instru_folder):
            for root, dirs, files in os.walk("{}/{}".format(folder, instru)):
                for erase in files:
                    os.remove(os.path.join(root, erase))

    for instru in os.listdir(folder):
        instru_sample = glob.glob('london_phill_dataset_multi/{}/*.wav'.format(instru))
        compt = 1
        for i in tqdm(range(len(instru_sample))):
            print("Processing {}".format(instru_sample[i]))
            y, sr = librosa.load(instru_sample[i])
            y_true = []
            for i in y:
                if abs(i) > 10 ** (-4):
                    y_true.append(i)
            librosa_name = "chunk{0}.wav".format(compt)
            sf.write("librosa/{}/{}".format(instru, librosa_name), np.array(y_true), sr)
            read_path = "librosa/{}/{}".format(instru, librosa_name)
            myaudio = AudioSegment.from_wav(read_path)
            chunk_length_ms = duration  # pydub calculates in millisec
            chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec
            # Export all of the individual chunks as wav files
            for i, chunk in enumerate(chunks):
                chunk_name = "chunk{0}.wav".format(i)
                chunk.export("{}/{}/{}".format(folder, instru, chunk_name), format="wav")
            compt += 1

    for instru in os.listdir(folder):
        instru_sample = glob.glob('london_phill_dataset_multi/{}/*.wav'.format(instru))
        compt = 1
        for i in tqdm(range(len(instru_sample))):
            librosa_name = "chunk{}.wav".format(compt)
            read_path = "librosa/{}/{}".format(instru, librosa_name)
            myaudio = AudioSegment.from_wav(read_path)
            chunk_length_ms = duration  # pydub calculates in millisec
            chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec
            # Export all of the individual chunks as wav files
            for i, chunk in enumerate(chunks):
                chunk_name = "chunk{}.wav".format(i)
                chunk.export("{}/{}/{}".format(folder, instru, chunk_name), format="wav")
            compt += 1



def create_database(file, folder_all_data):
    compt = 0
    with open(file, "w", newline='') as f:
        csv_file = csv.writer(f)
        data_labels = ["chroma_stft", "spectral_centroid", "spectral_bandwith", "rolloff",
                       "zero_crossing_rate"]
        data_labels.append("Instrument")
        csv_file.writerow(data_labels)
        instruments = os.listdir(folder_all_data)
        for i in instruments:
            all_samples = glob.glob('{}/{}/*.wav'.format(folder_all_data, i))
            print("Processing : {}".format(i))
            for sample in tqdm(all_samples):
                y, sr = librosa.load(sample, mono=True)
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                if chroma_stft != 0 and spec_cent != 0 and spec_bw != 0 and rolloff != 0 and zcr != 0:
                    data_list = [chroma_stft, spec_cent, spec_bw, rolloff, zcr]
                    data_list.append(i)
                    csv_file.writerow(data_list)
            compt += 1


def encode_csv(filename):
    data = pd.read_csv(filename, index_col=None, header=0, delimiter=',')
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    scaler = StandardScaler()
    x = scaler.fit_transform(np.array(data.iloc[:, :-1]))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, y_train, x_test, y_test

def create_neural_network(x_train, y_train, x_test, y_test, save, model_name):
    """Creation d'un réseau de neuronnes"""
    model = models.Sequential()
    nb_neuronnes = 128
    model.add(layers.Dense(nb_neuronnes, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(nb_neuronnes, activation='relu'))
    model.add(layers.Dense(nb_neuronnes, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=100,
                        batch_size=8)
    if save == True:
        model.save(model_name)

    true = 0
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == y_test[i]:
            true += 1
    print(true/len(y_pred))


    return model


def analyze(file, model, frame_duration):
    sample_number = 0
    for root, dirs, files in os.walk("PredictTest"):
        for erase in files:
            os.remove(os.path.join(root, erase))

    y, sr = librosa.load(file)
    y_true = []
    for i in y:
        if abs(i) > 10 ** (-3):
            y_true.append(i)
    sf.write("PredictTest/file_erase.wav", np.array(y_true), sr)
    myaudio = AudioSegment.from_file("PredictTest/file_erase.wav", "wav")
    chunk_length_ms = frame_duration  # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec

    # Export all of the individual chunks as wav files
    print("Creating frames \n")
    for i in tqdm(range(len(chunks))):
        chunk_name = "chunk{0}.wav".format(i)
        chunks[i].export("PredictTest/{}".format(chunk_name), format="wav")

    All_chunks = glob.glob("PredictTest/*.wav")

    with open("PredictTest/PredictData.csv", "w", newline='') as f:
        csv_file = csv.writer(f)
        data_labels = ["chroma_stft", "spectral_centroid", "spectral_bandwith", "rolloff", "zero_crossing_rate"]
        csv_file.writerow(data_labels)
        print("\n Creating database \n")
        for j in tqdm(range(len(All_chunks))):
            y, sr = librosa.load(All_chunks[j], mono=True)
            chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            if chroma_stft != 0 and spec_cent != 0 and spec_bw != 0 and rolloff != 0 and zcr != 0:
                sample_number += 1
                data_list = [chroma_stft, spec_cent, spec_bw, rolloff, zcr]
                csv_file.writerow(data_list)

    data = pd.read_csv('PredictTest/PredictData.csv', index_col=None, header=0, delimiter=',')
    scaler = StandardScaler()
    x_predict = scaler.fit_transform(np.array(data.iloc[:, :]))
    prediction = model.predict(x_predict)

    list_result = [0 for i in range(6)]
    print("\n Analyzing data \n")
    for i in tqdm(range(sample_number)):
        result = np.argmax(prediction[i])
        list_result[result] += 1
    print(list_result)
    instruments = ["cello", "flute", "oboe", "sax", "trumpet", "viola"]
    for i in range(len(list_result)):
        if list_result[i] == max(list_result):
            print(instruments[i])


"""
Used for data visualization and research

def get_features_from_data(folder_all_data):
    data_cello, data_flute, data_oboe, data_sax, data_trumpet, data_viola = [], [], [], [], [], []
    instruments = ["cello", "flute", "oboe", "sax", "trumpet", "viola"]
    for instru in instruments:
        all_sax = glob.glob('{}/{}'.format(folder_all_data, instru))
        print("Processing : {}".format(instru))
        sax_sample = []
        for sax in tqdm(range(1, 101)):
            one_sax = glob.glob('{}/{}/{}/*.wav'.format(folder_all_data, instru, sax))
            for file in one_sax:
                y, sr = librosa.load(file, mono=True)
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                if chroma_stft != 0 and spec_cent != 0 and spec_bw != 0 and rolloff != 0 and zcr != 0:
                    data_list = [chroma_stft, spec_cent, spec_bw, rolloff, zcr]
                    if instru == "cello":
                        data_cello.append(data_list)
                    elif instru == "flute":
                        data_flute.append(data_list)
                    elif instru == "oboe":
                        data_oboe.append(data_list)
                    elif instru == "sax":
                        data_sax.append(data_list)
                    elif instru == "trumpet":
                        data_trumpet.append(data_list)
                    elif instru == "viola":
                        data_viola.append(data_list)

    with open("data/cello_data", "wb") as f:
        data_cello = pickle.dump(data_cello, f)
    with open("data/flute_data", "wb") as f:
        data_flute = pickle.dump(data_flute, f)
    with open("data/oboe_data", "wb") as f:
        data_oboe = pickle.dump(data_oboe, f)
    with open("data/sax_data", "wb") as f:
        data_sax = pickle.dump(data_sax, f)
    with open("data/trumpet_data", "wb") as f:
        data_trumpet = pickle.dump(data_trumpet, f)
    with open("data/viola_data", "wb") as f:
        data_viola = pickle.dump(data_viola, f)

def plot_feature(x, y):
    instruments = ["cello", "flute", "oboe", "sax", "trumpet", "viola"]
    data = []
    with open("data/cello_data", "rb") as file_cello:
        data_cello = pickle.load(file_cello)
        X1, Y1 = [], []
        for i in data_cello[1:int(len(data_cello) / 10)]:
            X1.append(i[x])
            Y1.append(i[y])
        plt.scatter(X1, Y1, c='b', label='cello', marker='s')

    with open("data/flute_data", "rb") as file_flute:
        data_flute = pickle.load(file_flute)
        X2, Y2 = [], []
        for i in data_flute[1:int(len(data_flute) / 10)]:
            X2.append(i[x])
            Y2.append(i[y])
        plt.scatter(X2, Y2, c='g', label='flute', marker='o')

    with open("data/oboe_data", "rb") as file_oboe:
        data_oboe = pickle.load(file_oboe)
        X3, Y3 = [], []
        for i in data_oboe[1:int(len(data_oboe) / 10)]:
            X3.append(i[x])
            Y3.append(i[y])
        plt.scatter(X3, Y3, c='r', label='oboe', marker='x')

    with open("data/sax_data", "rb") as file_sax:
        data_sax = pickle.load(file_sax)
        X4, Y4 = [], []
        for i in data_sax[1:int(len(data_sax) / 10)]:
            X4.append(i[x])
            Y4.append(i[y])
        plt.scatter(X4, Y4, c='y', label='sax', marker='+')

    with open("data/trumpet_data", "rb") as file_trumpet:
        data_trumpet = pickle.load(file_trumpet)
        X5, Y5 = [], []
        for i in data_trumpet[1:int(len(data_trumpet) / 10)]:
            X5.append(i[x])
            Y5.append(i[y])
        plt.scatter(X5, Y5, c='magenta', label='trumpet', marker='+')

    with open("data/viola_data", "rb") as file_viola:
        data_viola = pickle.load(file_viola)
        X6, Y6 = [], []
        for i in data_viola[1:int(len(data_viola) / 10)]:
            X6.append(i[x])
            Y6.append(i[y])
        plt.scatter(X6, Y6, c='black', label='trumpet', marker='*')

    plt.show()
    
"""

"""
Need to create samples (25 ms) from all files to perform analysis. Do it one time after samples are saved in
#the folder specified. A folder librosa is created for calculation uses
You need to create the folder you specified then create directory for each instrument you want to use like it is done
in the folder london_phil_dataset_multi 
"""
partitionnate("data_partitionated", 25)

#Create a csv file from the samples created, do it one time then use the csv
#create_database("csv/data.csv", "data_partitionated/")


#Encode = encode_csv("csv/data.csv")
#x_train = Encode[0]
#y_train = Encode[1]
#x_test = Encode[2]
#y_test = Encode[3]

#The model can be saved and reused after
#model = create_neural_network(x_train, y_train, x_test, y_test, True, 'Test')
model = load_model('Test')

"""
Analyse the wav file specifying the model and the millisecond used to partitionate
You need to create a PredictTest folder in the root of the project where data will be save to analyse the file
"""
analyze("tenor-sax_95bpm_G_major.wav", model, 25)


