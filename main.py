import numpy as np
from file_reader import read_edf, text_reader
from filter import fe_freqBandMean, fe_wavelet


fn = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01_03.edf"
summary_txt = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01-summary.txt"

seiz_dict, fs = text_reader(summary_txt)

eeg_data = read_edf(fn)

i = 0
# for key in seiz_dict:

key = 'chb01_04.edf'
    #assume only 1 seizure for now
start_time = int(seiz_dict[key]['start_time'][0])
end_time   = int((seiz_dict[key]['end_time'][0]))

seiz_data = eeg_data[:, start_time*fs:end_time *fs]

# freq_band = fe_freqBandMean(seiz_data, fs)
wavelet_decomp = fe_wavelet(seiz_data, fs)

print(seiz_data.shape, wavelet_decomp.shape)

