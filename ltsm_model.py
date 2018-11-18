from file_reader import read_edf, text_reader
import numpy as np
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
import scipy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from filter import fe_wavelet

smoothing_window_size = 2 * 256 #2 second windom

## tranform time series data set to supervised learning


def series_to_supervised(data, n_in=1, n_out =1, dropnan=True):
    """

    :param data: Eeg data. 1 electrode
    :param n_in:  Number of lag observations as input
    :param n_out:  Numbers of observations as output
    :param dropnan:  Whether to drop nan rows
    :return:
    """


    n_vars = data.shape[1]


def find_energy(data, window_size = 256):
    N = len(data)

    window = np.hamming(window_size)
    window.shape = (window_size,1)
    print(window.shape)

    n = N - window_size #Number of windowed samples

    print(n)

    p = np.power(data, 2)
    print(p.shape)
    # s = stride_tricks.as_strided((p, shape=(n, window_size), stirdes = (data.item)))
    # e = np.dot( s, window)/ window_size
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def create_model(X_train, y_train, X_test, y_test):
    # tot_len = X_train + X_test
    # embedding_vector_length = 32
    # max_review_length = 512 #2 seconds

    model = Sequential()
    # model.add(Embedding(tot_len, embedding_vector_length, input_length=max_review_length))
    # model.add(LSTM(100))
    model.add(LSTM(100, input_shape=1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


def evaluate_simple_model(X_train, y_train, X_test, y_test, look_back=1):
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
	_, accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
	print(accuracy)

def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

def evaluate_bi_model(X_train, y_train, X_test, y_test):
	model = Sequential()
	model.add(Bidirectional(LSTM(4, input_shape=(1, look_back))))
	model.add(Dense(2, activation='softmax'))
	# model.compile(loss='mean_squared_error', optimizer='adam')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
	_, accuracy = model.evaluate(testX, testY, batch_size=32, verbose=0)

#
# root_folder = "/Users/graemecox/Documents/ChildSeizure/chb01/"
# summary_txt = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01-summary.txt"
#
# seiz_dict, fs = text_reader(summary_txt)
#
# key = 'chb01_03.edf'
#
# eeg_data = read_edf(root_folder+key)
#
# start_ind = int(seiz_dict[key]['start_time'][0]) * fs
# end_ind = int(seiz_dict[key]['end_time'][0]) *fs
#
# seiz_data = eeg_data[:, start_ind: end_ind]
# # print(seiz_data.shape)
#
# c4_P4 = eeg_data[9,:]
#
# feat = fe_wavelet(eeg_data, fs)
# print(feat.shape)
#
# # seiz_elec = seiz_data[9,:]
# X_seiz = c4_P4[ start_ind: end_ind]
# y_seiz = np.ones(X_seiz.shape[0])
#
# test_pre = c4_P4[start_ind - 5000 : start_ind]
# test_post= c4_P4[end_ind: end_ind + 5000]
#
# # X_new = np.reshape(seiz_elec, (-1,))
# # print(seiz_elec.shape, X_new.shape)
#
# #
# # print(X_seiz.shape)
# # print(test_pre.shape)
# # print(test_post.shape)
#
# X_non_seiz = np.append(test_pre, test_post, axis=0)
# y_non_seiz = np.ones(X_non_seiz.shape[0])
#
# X_pre_shuf = np.append(X_non_seiz, X_seiz, axis=0)
# y_pre_shuf = np.append(y_non_seiz, y_seiz, axis=0)
# # print(X_pre_shuf.shape)
# # print(y_pre_shuf.shape)
#
# X = X_pre_shuf
# y = y_pre_shuf
#
# # create_model(X_train, y_train, X_test, y_test)
#
# print(X.shape, y.shape)
#
# # X = fe_wavelet(X)
#
# # X_train, X_test, y_train, y_test = train_test_split(X_pre_shuf, y_pre_shuf, test_size=0.30, shuffle=True)
# # print(X_train.shape, y_train.shape)
# # print(X_test.shape, y_test.shape)
#
# def evaluate_model(trainX, trainy, testX, testy):
# 	verbose, epochs, batch_size = 0, 15, 64
# 	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
# 	model = Sequential()
# 	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(100, activation='relu'))
# 	model.add(Dense(n_outputs, activation='softmax'))
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# fit network
# 	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# 	# evaluate model
# 	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
# 	return accuracy

# evaluate_model(X_train, y_train, X_test, y_test)
#
# model = Sequential()
# # model.add(Embedding(tot_len, embedding_vector_length, input_length=max_review_length))
# # model.add(LSTM(100))
# model.add(LSTM(100, input_shape=X_train.shape, return_sequences=True))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=1, batch_size=64)
# print(model.summary())
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1] * 100))
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
