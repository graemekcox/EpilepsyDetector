import numpy as np
from file_reader import read_edf, text_reader
from filter import fe_freqBandMean, fe_wavelet
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from prep_data import get_data_epochs, append_epochs

# fn = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01_03.edf"
root_folder = "/Users/graemecox/Documents/ChildSeizure/chb01/"
summary_txt = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01-summary.txt"

seiz_dict, fs = text_reader(summary_txt)

# eeg_data = read_edf(fn)

i = 0

# key = 'chb01_03.edf'

##C4-P4 (channel 10) T8-p8 (channel 15)

##### Prep all input data
window_size = 2 * fs
X = np.empty((0, window_size))
labels = np.empty((0))

for key in seiz_dict.keys():
    print('-----------------' + key + '----------------')

    fn = root_folder+key
    eeg_data = read_edf(fn)

    start_time = int(seiz_dict[key]['start_time'][0])
    end_time = int(seiz_dict[key]['end_time'][0])


    c4_p4 = eeg_data[9,:]
# print('-------Original shape--------')
# print(c4_p4.shape)
# print('-------Start adding other shapes------')




    temp_x, temp_labels = append_epochs(c4_p4, start_time, end_time, fs)


    # epochs = np.append(epochs, temp_labels, axis=0)
    # labels = np.append(labels, temp_labels, axis=0)
    X = np.append(X, temp_x, axis=0)
    labels = np.append(labels, temp_labels, axis=0)

    print(X.shape)

    # X = np.append()


print('-------- Final Size ---------')
print(X.shape, labels.shape)


print('--------- Unique values ---------')
unique, counts = np.unique(labels, return_counts=True)
print(np.asarray((unique, counts)).T)

np.save('X.npy', X)
np.save('labels.npy',labels)