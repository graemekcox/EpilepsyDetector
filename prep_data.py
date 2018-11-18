import numpy as np

def get_data_epochs(data, epoch_size, is_seizure=0):
    """
    Split up data into multiple small segmenets

    """

    i = 0
    num_epoch = int(len(data) / epoch_size)  # round down

    epochs = np.empty(( 0,epoch_size))

    for i in range(num_epoch):
        epochs = np.append(epochs, [data[i*epoch_size: (i+1)*epoch_size]], axis=0)

    if (is_seizure):
        y = np.zeros(epochs.shape[0])
    else:
        y = np.ones(epochs.shape[0])

    # print(y.shape, epochs.shape)

    return epochs, y



def append_epochs(data, start_time, end_time, fs):
    epoch_size = fs*2

    epochs = np.empty((0, epoch_size))
    labels = np.empty((0))

    # add data before seizure
    temp_epochs, temp_y = get_data_epochs(data[:start_time * fs], epoch_size, 0)
    epochs = np.append(epochs, temp_epochs, axis=0)
    labels = np.append(labels, temp_y, axis=0)

    # add data after seizure
    temp_epochs, temp_y = get_data_epochs(data[end_time * fs:], epoch_size, 0)
    epochs = np.append(epochs, temp_epochs, axis=0)
    labels = np.append(labels, temp_y, axis=0)

    # add seizure data
    seiz_data = data[start_time * fs:end_time * fs]
    temp_epochs, temp_y = get_data_epochs(seiz_data, epoch_size, 1)
    epochs = np.append(epochs, temp_epochs, axis=0)
    labels = np.append(labels, temp_y, axis=0)

    print("appeneded epochs")
    return epochs, labels