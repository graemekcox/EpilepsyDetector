import numpy as np
from file_reader import read_edf, text_reader
from filter import fe_freqBandMean, fe_wavelet
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split


fn = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01_03.edf"
summary_txt = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01-summary.txt"

seiz_dict, fs = text_reader(summary_txt)

eeg_data = read_edf(fn)

i = 0

key = 'chb01_03.edf'
start_time = int(seiz_dict[key]['start_time'][0])
end_time = int(seiz_dict[key]['end_time'][0])



    #assume only 1 seizure for now
# start_time = int(seiz_dict[key]['start_time'][0])
# end_time   = int((seiz_dict[key]['end_time'][0]))
#
# seiz_data = eeg_data[:, start_time*fs:end_time *fs]

# freq_band = fe_freqBandMean(seiz_data, fs)


##C4-P4 (channel 10) T8-p8 (channel 15)

def get_non_seizure_data(data, start_time, end_time, fs):
    """
    Split up data into multiple small 2 second segmenets

    """
    #2 seconds
    epoch_size = 2*fs

    left_arr = data[ : start_time*fs]
    right_arr= data[ end_time*fs:]

    def get_epochs(data, epoch_size):
        i = 0
        num_epoch = int(len(data) / epoch_size)  # round down
        print("Num epoch " + str(num_epoch))
        epochs = np.empty(( 0,epoch_size))

        for i in range(num_epoch):
            epochs = np.append(epochs, [data[i*epoch_size: (i+1)*epoch_size]], axis=0)

        return epochs

    epochs = np.empty((0, epoch_size))

    epochs = np.append(epochs, get_epochs(left_arr, epoch_size), axis=0)
    epochs = np.append(epochs, get_epochs(right_arr, epoch_size), axis=0)

    y = np.zeros(epochs.shape[0])
    print(y.shape, epochs.shape)

    return epochs, y




data = eeg_data[9,:]
print(data.shape)

epochs, y = get_non_seizure_data(data, start_time, end_time, fs)




# X = np.empty((0, 5))
# for key in seiz_dict:
#
#
#     start_time = int(seiz_dict[key]['start_time'][0])
#     end_time = int((seiz_dict[key]['end_time'][0]))
#
#     seiz_data = eeg_data[:, start_time * fs:end_time * fs]
#     wavelet_decomp = fe_wavelet(seiz_data, fs)
#
#     #append features to feature vector
#     X = np.append(X, wavelet_decomp, axis=0)
#
# y = np.ones((X.shape[0],1))
#

# print(X.shape, y.shape)

# clf = svm.SVC(kernel='rbf', C=1.0)
# clf.fit(X,y)
#
# y_pred = clf.predict(X_test)

