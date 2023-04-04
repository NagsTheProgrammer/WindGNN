import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def __create_sequences(data, seq_length):
    xs, ys = [], []

    r = int((len(data) - len(data) % seq_length) / seq_length)

    for i in range(r):
        x = data[(i * seq_length):((i + 1) * seq_length), :, 2:15]
        y1 = data[(i * seq_length + 1):((i + 1) * seq_length + 1), :, 13]
        y2 = data[(i * seq_length + 2):((i + 1) * seq_length + 2), :, 13]
        y3 = data[(i * seq_length + 3):((i + 1) * seq_length + 3), :, 13]
        ym = np.concatenate((y1, y2), axis=1)
        y = np.concatenate((ym, y3), axis=1)
        xs.append(x)
        ys.append(y)
    xs1 = np.array(xs)
    ys1 = np.array(ys)
    indices = np.arange(xs1.shape[0])
    np.random.shuffle(indices)
    xs1 = xs1[indices]
    ys1 = ys1[indices]
    return xs1, ys1


def generate_sequences(df, batch_size, device):
    # plan, take in df, refactor into samples of 168 hourly timesteps in 3-dim tensor (x, y, z) = [timestep, station, feature]
    attr_matrix = df
    num_stations = 0
    num_attr = df.shape[1] - 2

    attr_matrix_transpose = []
    initial = True
    stations = np.unique(attr_matrix[:, 0])  # get list of unique station values
    for station in stations:
        num_stations += 1
        station_matrix = attr_matrix[np.where(attr_matrix[:, 0] == station)]
        station_matrix = station_matrix.reshape((station_matrix.shape[0], 1, station_matrix.shape[1]))
        if initial:
            attr_matrix_transpose = station_matrix
            initial = False
        else:
            attr_matrix_transpose = np.concatenate((attr_matrix_transpose, station_matrix), axis=1)

    # Split the data into train and test sets, shuffling
    train_df, test_df = train_test_split(attr_matrix_transpose, test_size=0.3, shuffle=False)

    # Create sequences
    seq_length = 168  # 168-hour sequences
    train_sequences, train_labels = __create_sequences(train_df, seq_length)
    test_sequences, test_labels = __create_sequences(test_df, seq_length)

    # Convert sequences to tensors
    train_sequences = torch.tensor(train_sequences.tolist(), dtype=torch.float).to(device)
    train_labels = torch.tensor(train_labels.tolist(), dtype=torch.float).to(device)
    test_sequences = torch.tensor(test_sequences.tolist(), dtype=torch.float).to(device)
    test_labels = torch.tensor(test_labels.tolist(), dtype=torch.float).to(device)

    # Getting loaders and epochs
    train_dataset = TensorDataset(train_sequences, train_labels)
    test_dataset = TensorDataset(test_sequences, test_labels)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader, num_attr, num_stations
