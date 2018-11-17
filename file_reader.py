import pyedflib
import numpy as np
import matplotlib.pyplot as plt

fn = "/Users/graemecox/Documents/ChildSeizure/chb01/chb01_03.edf"

def read_edf(fn):
    f = pyedflib.EdfReader(fn)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    data = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        data[i, :] = f.readSignal(i)

    fs = 256  # Hz

    print("Length = " + str(1 / fs * data.shape[1]))

    print(data.shape)

    seiz_start = 2996
    seiz_end = 3036
    seiz_data = data[:, seiz_start * fs:seiz_end * fs]
    print(seiz_data.shape)


read_edf(fn)
# plt.subplot(2,2,1)
# plt.plot(seiz_data[0,:])
#
# plt.subplot(2,2,2)
# plt.plot(seiz_data[1,:])
#
# plt.subplot(2,2,3)
# plt.plot(seiz_data[2,:])
#
# plt.subplot(2,2,4)
# plt.plot(seiz_data[3,:])

# plt.show()
# plt.subplot(2,2,1)
# plt.plot(sigbufs[1,0:1000])
# #
#
# plt.subplot(2,2,2)
# plt.plot(sigbufs[2,0:1000])
#
# plt.show()