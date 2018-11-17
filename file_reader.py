import pyedflib
import numpy as np
import matplotlib.pyplot as plt

def read_edf(fn):
    f = pyedflib.EdfReader(fn)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    data = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        data[i, :] = f.readSignal(i)



    # print("Length = " + str(1 / int(fs) * data.shape[1]))

    print(data.shape)
    #
    # seiz_start = 2996
    # seiz_end = 3036
    # seiz_data = data[:, seiz_start * fs:seiz_end * fs]
    # print(seiz_data.shape)
    return data


def text_reader(fn):

    lines = [line.rstrip('\n') for line in open(fn)]

    seizure_dict = {}
    patient_dict = {}

    # for line in lines:
    line_ind = 0


    while line_ind < len(lines):

        line = lines[line_ind]
        # print(line)
        if "Data Sampling Rate" in line:
            fs = int(line.split(': ')[1].split(' ')[0])


        if 'File Name' in line:
            id = line.split(': ')[1]
            patient_dict[id] = {'start_time':0, 'end_time':0, 'num_seiz':0}
            # print(patient_dict[id])

            line_ind += 1
            line = lines[line_ind]
            # Set end time
            patient_dict[id]['start_time'] = line.split(': ')[1]

            line_ind += 1
            line = lines[line_ind]
            # Set end time
            patient_dict[id]['end_time'] = line.split(': ')[1]

            line_ind += 1
            line = lines[line_ind]
            # print(line)
            #
            # # Get number of seiz
            seiz_num = int(line.split(': ')[1])

            if (seiz_num != 0):
                print('Seziure in id = '+id)

                seizure_dict[id] = {'start_time': [], 'end_time': []}
                for i in range(seiz_num):

                    line_ind += 1
                    line = lines[line_ind]

                    seizure_dict[id]['start_time'].append(line.split(': ')[1].split(' ')[0])

                    line_ind += 1
                    line = lines[line_ind]

                    seizure_dict[id]['end_time'].append(line.split(': ')[1].split(' ')[0])

        line_ind += 1
        # def check_file(line):
    return seizure_dict, fs








# read_edf(fn)
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