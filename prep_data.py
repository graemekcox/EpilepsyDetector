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

    print(y.shape, epochs.shape)

    return epochs, y

