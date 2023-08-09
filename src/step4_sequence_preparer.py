import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def __create_sequences(data, seq_length):
    xs, ys = [], []

    r = int((len(data) - len(data) % seq_length) / seq_length)

    for i in range(r-1):
        x = data[(i * seq_length):((i + 1) * seq_length), :, 2:17] # 2:17 for OG paper, 2:4 for wind only, 2:6 for wind and direction
        y1 = data[(i * seq_length + 1):((i + 1) * seq_length + 1), :, 13] # 13 for OG paper, 3 for wind only, 5 for wind and direction
        y2 = data[(i * seq_length + 2):((i + 1) * seq_length + 2), :, 13] # "   "  "    "    "  "   "    "
        y3 = data[(i * seq_length + 3):((i + 1) * seq_length + 3), :, 13] # "   "  "    "    "  "   "    "
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
    
    # Create sequences
    seq_length = 168  # 168-hour sequences

    # Split the data into train and test sets
    temp_df, test_df = train_test_split(attr_matrix_transpose, test_size=0.2, shuffle=False)
    dev1, dev2 = train_test_split(temp_df, test_size=0.5, shuffle=False)
    train1, val1 = train_test_split(dev1, test_size=0.2, shuffle=False)
    train2, val2 = train_test_split(dev2, test_size=0.2, shuffle=False)

    train1_sequence, train1_label = __create_sequences(train1, seq_length)
    train2_sequence, train2_label = __create_sequences(train2, seq_length)
    train_sequences = np.concatenate((train1_sequence, train2_sequence))
    train_labels = np.concatenate((train1_label, train2_label))
    
    val1_sequence, val1_label = __create_sequences(val1, seq_length)
    val2_sequence, val2_label = __create_sequences(val2, seq_length)
    validate_sequences = np.concatenate((val1_sequence, val2_sequence))
    validate_labels = np.concatenate((val1_label, val2_label))

    test_sequences, test_labels = __create_sequences(test_df, seq_length)

    # Convert sequences to tensors
    train_sequences = torch.tensor(train_sequences.tolist(), dtype=torch.float).to(device)
    train_labels = torch.tensor(train_labels.tolist(), dtype=torch.float).to(device)

    validate_sequences = torch.tensor(validate_sequences.tolist(), dtype=torch.float).to(device)
    validate_labels = torch.tensor(validate_labels.tolist(), dtype=torch.float).to(device)

    test_sequences = torch.tensor(test_sequences.tolist(), dtype=torch.float).to(device)
    test_labels = torch.tensor(test_labels.tolist(), dtype=torch.float).to(device)

    # Getting loaders and epochs
    train_dataset = TensorDataset(train_sequences, train_labels)
    validate_dataset = TensorDataset(validate_sequences, validate_labels)
    test_dataset = TensorDataset(test_sequences, test_labels)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                             batch_size=1,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=True)
    

    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False)

    return train_loader, train_loader2, val_loader, test_loader, num_attr, num_stations
