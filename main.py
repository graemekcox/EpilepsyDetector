import numpy as np
from file_reader import read_edf, text_reader
from filter import fe_freqBandMean, fe_wavelet
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from prep_data import get_data_epochs

fn = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01_03.edf"
summary_txt = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01-summary.txt"

seiz_dict, fs = text_reader(summary_txt)

eeg_data = read_edf(fn)

i = 0

key = 'chb01_03.edf'
start_time = int(seiz_dict[key]['start_time'][0])
end_time = int(seiz_dict[key]['end_time'][0])

##C4-P4 (channel 10) T8-p8 (channel 15)

##### Prep all input data


c4_p4 = eeg_data[9,:]
print('-------Original shape--------')
print(c4_p4.shape)

print('-------Start adding other shapes------')
epoch_size = 2*fs




epochs = np.empty((0, epoch_size))
labels = np.empty((0))

# add data before seizure
temp_epochs, temp_y = get_data_epochs(c4_p4[:start_time*fs], epoch_size, 0)
epochs = np.append(epochs, temp_epochs, axis=0)
labels = np.append(labels, temp_y, axis=0)

# add data after seizure
temp_epochs, temp_y = get_data_epochs(c4_p4[end_time*fs:], epoch_size, 0)
epochs = np.append(epochs, temp_epochs, axis=0)
labels = np.append(labels, temp_y, axis=0)

# add seizure data
seiz_data = c4_p4[start_time*fs:end_time *fs]
temp_epochs, temp_y = get_data_epochs(seiz_data, epoch_size, 1)
epochs = np.append(epochs, temp_epochs, axis=0)
labels = np.append(labels, temp_y, axis=0)


print(epochs.shape, labels.shape)
