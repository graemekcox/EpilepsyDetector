
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pywt


freq_bands = {'Delta': (0,4),
			'Theta': (4,8),
			'Alpha': (8,12),
			'Beta': (12,30),
			'Gamma': (30,45)
			}

def fe_freqBandMean(data, Fs):
	# amps = np.absolute(np.fft.rfft(data))
	# freqs = np.fft.rfftfreq(len(data), 1.0/Fs)
	#
	# temp_band = dict()
	# for band in freq_bands:
	# 	i = np.where((freqs >= freq_bands[band][0]) &
	# 				 (freqs <= freq_bands[band][1]))[0]
	# 	temp_band[band] = np.mean(amps[i])
	#
	#
	# 	values = [temp_band['Delta'],
	# 			  temp_band['Theta'],
	# 			  temp_band['Alpha'],
	# 			  temp_band['Beta'],
	# 			  temp_band['Gamma']]
	amps = np.absolute(np.fft.rfft(data))
	freqs = np.fft.rfftfreq(len(data), 1.0 / Fs)

	# Take the mean of the fft amplitude for each EEG band
	temp_band = dict()
	for band in freq_bands:
		# Find all indexs that belong to each EEG frequency band
		i = np.where((freqs >= freq_bands[band][0]) &
					 (freqs <= freq_bands[band][1]))[0]
		temp_band[band] = np.mean(amps[i])

	# print(L)
	# features = [][]
	values = [temp_band['Delta'],
			  temp_band['Theta'],
			  temp_band['Alpha'],
			  temp_band['Beta'],
			  temp_band['Gamma']]

	return np.reshape(np.array(values), (-1,5)) #return 2d array


def mean(data):
	return sum(data)/float(len(data))

def quickFFT(data):
	Y = np.fft.fft(data)
	L = int(len(data))

	P2 = np.abs(Y/L)

	f_seiz = P2[0:int(L/2+1)]
	f_seiz[1:int(L/2+1)] = 2*f_seiz[1:int(L/2+1)]
	return f_seiz

def fe_wavelet(data, Fs):
	num_elec = data.shape[0]

	wname = 'db4';

	w = pywt.Wavelet(wname)
	features = []
	for i in range(num_elec):
		elec_data = data[i][:]

		filt = np.convolve(elec_data, w.dec_lo)
		filt_D4 = np.convolve(filt, w.dec_hi)

		f_seiz = quickFFT(filt_D4)
		# print(L)
		# features = [][]
		values = [mean(f_seiz[49:75]),
			mean(f_seiz[75:100]),
			mean(f_seiz[124:150]),
			mean(f_seiz[149:175]),
			mean(f_seiz[174:200])]

		features.append(values)
		# print(np.array(values).shape)


	features = np.array(features)
	return features
