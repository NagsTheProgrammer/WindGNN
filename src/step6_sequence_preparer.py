import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def __create_sequences(data, seq_length):
    xs, ys = [], []

    r = int(( len(data) - len(data) % seq_length ) / seq_length)

    for i in range(r):
        x = data[( i*seq_length ):( ( i + 1 ) * seq_length ), :, 2:15]
        y = data[( i*seq_length + 1 ):( ( i + 1 ) * seq_length + 1 ), :, 13]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def generate_sequences(df, batch_size):
    # plan, take in df, refactor into samples of 168 hourly timesteps in 3-dim tensor (x, y, z) = [timestep, station, feature]
    # df before looks like 2-dim matrix (x, y) = [feature, timestep] *** includes all stations

    attr_matrix = df.to_numpy()
    # attr_matrix = np.array_split(df, 7) # splitting df into each station *** works for small dataset, may not work for larger dataset as total timesteps for each station is unbalanced
    # dLength = len(attr_matrix[0])

    attr_matrix_transpose = []
    initial = True
    stations = np.unique(attr_matrix[:, 0]) # get list of unique station values
    for station in stations:
        station_matrix = attr_matrix[np.where(attr_matrix[:,0] == station)]
        station_matrix = station_matrix.reshape((station_matrix.shape[0], 1, station_matrix.shape[1]))
        if initial:
            attr_matrix_transpose = station_matrix
            initial = False
        else:
            attr_matrix_transpose = np.concatenate((attr_matrix_transpose, station_matrix), axis = 1)
        # attr_matrix_transpose = np.append(attr_matrix_transpose, np.transpose(attr_matrix[(attr_matrix[:, 0] == station)])) # finds matrix subset with single station, transposes, and appends to attr_matrix_transpose

    print(attr_matrix_transpose.shape)
    # if above didn't work, this should
    # attr_matrix_transpose = []
    # for a in attr_matrix:
    #     attr_matrix_transpose = np.append(attr_matrix_transpose, np.transpose(a)) # transposing into [feature, timestep]

    # attr_matrix_stack = np.stack(attr_matrix_transpose)

    # print(attr_matrix_stack.shape) # printing shape - expected (7, 13, 184107) = [station, faeture, timestep]

    # Get the output of the GCN
    # gcn_output = model(data).detach().numpy() # Detaching result from computation graph, converting into a numpy array

    # Prepare the data for sequence generation
    # df = df.set_index("Date/Time") # Setting index of df to Date/Time
    # df = df.loc[:, gcn_output.columns] # Getting only columns from gcn_output
    # df["Wind Speed 10 m Avg. (km/h)"] = gcn_output.squeeze() # Setting wind speed column of df to scalar of gcn_output

    # Split the data into train and test sets, shuffling
    train_df, test_df = train_test_split(attr_matrix_transpose, test_size=0.3, shuffle=False)

    # Create sequences
    seq_length = 168 # 168-hour sequences
    train_sequences, train_labels = __create_sequences(train_df, seq_length)
    test_sequences, test_labels = __create_sequences(test_df, seq_length)

    # Convert sequences to tensors
    train_sequences = torch.tensor(train_sequences.tolist(), dtype=torch.float)
    train_labels = torch.tensor(train_labels.tolist(), dtype=torch.float)
    test_sequences = torch.tensor(test_sequences.tolist(), dtype=torch.float)
    test_labels = torch.tensor(test_labels.tolist(), dtype=torch.float)

    # Getting loaders and epochs
    train_dataset = TensorDataset(train_sequences, train_labels)
    test_dataset = TensorDataset(test_sequences, test_labels)

    # train_loader = torch.utils.data.DataLoader(dataset=train_sequences,
    #                                            batch_size=batch_size,
    #                                            shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test_sequences,
    #                                           batch_size=batch_size,
    #                                           shuffle=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader