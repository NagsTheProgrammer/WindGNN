import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from step1_loading_preprocessing import *
from step2_graph_builder import *
from step3_feature_extractor import *
from step4_gcn_layer_model import *
from step6_sequence_preparer import *
from step5_gcn_gru_combined_model import *
# from step8_trainer import *

if __name__ == "__main__":
    # Step 1a - Load data from both csv (measurements and coordinates)
    # Step 1b - Preprocess data
    df = load_and_process_wind_speed_dataset()

    # Step 2 - Build distance graph
    adj_matrix = build_graph(df)
    dLength = len(adj_matrix)

    # Step 3 - Select important features
    attr_matrix = extract_features(df)
    # attr_matrix = np.array_split(extract_features(df), dLength)
    train_loader, test_loader = generate_sequences(attr_matrix)

    # Step 4 - Define GCN

    # Defining Adjacency Matrix, Attribute Matrix, Ground Truth
    # adj_matrix -> manually derived
    # attr_matrix -> output from feature extractor
    # ground_truth -> basically y_train(?)
    # *** figure out where to get these from
    # adj_matrix = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    # attr_matrix = torch.randn(3, 4)
    # ground_truth = torch.randn(3, 2)
    # ground_truth = attr_matrix

    # Step 4,5 - Defining Two-Layer GCN
    model = GCN_GRU(input_dim=13, hidden_dim=13, output_dim=13, gru_hidden_dim=13) # Pretty sure all our dimensions are the same, need to confirm

    # Using GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Step 5 - Generate Train / Test Data Sequences

    n_iters = 100 # arbitrarily chosen, one cycle of
    batch_size = 64 # arbitrarily chosen
    train_loader, test_loader = generate_sequences(df, model, batch_size)

def __create_sequences(data, seq_length):
    xs, ys = [], []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length), :]
        y = data[(i + seq_length), :]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def generate_sequences(df, batch_size):
    # plan, take in df, refactor into samples of 168 hourly timesteps in 3-dim tensor (x, y, z) = [station, feature, timestep]
    # df before looks like 2-dim matrix (x, y) = [feature, timestep] *** includes all stations

    attr_matrix = np.array_split(df, 7) # splitting df into each station *** works for small dataset, may not work for larger dataset as total timesteps for each station is unbalanced
    dLength = len(attr_matrix[0])

    attr_matrix_transpose = []
    station_col = np.where(attr_matrix == "Verger AGCM")[0][0]
    stations = np.unique(attr_matrix[:, station_col]) # get list of unique station values
    for station in stations:
        attr_matrix_transpose = np.append(attr_matrix_transpose, np.transpose(attr_matrix[(attr_matrix[:, station_col] == station)])) # finds matrix subset with single station, transposes, and appends to attr_matrix_transpose

    # if above didn't work, this should
    # attr_matrix_transpose = []
    # for a in attr_matrix:
    #     attr_matrix_transpose = np.append(attr_matrix_transpose, np.transpose(a)) # transposing into [feature, timestep]

    attr_matrix_stack = np.stack(attr_matrix_transpose)

    print(attr_matrix_stack.shape) # printing shape - expected (7, 13, 184107) = [station, faeture, timestep]

    # Get the output of the GCN
    # gcn_output = model(data).detach().numpy() # Detaching result from computation graph, converting into a numpy array

    # Prepare the data for sequence generation
    df = df.set_index("Date/Time") # Setting index of df to Date/Time
    df = df.loc[:, gcn_output.columns] # Getting only columns from gcn_output
    df["Wind Speed 10 m Avg. (km/h)"] = gcn_output.squeeze() # Setting wind speed column of df to scalar of gcn_output

    # Split the data into train and test sets, shuffling
    train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True)

    # Create sequences
    seq_length = 168 # 168-hour sequences
    train_sequences, train_labels = __create_sequences(train_df.values, seq_length)
    test_sequences, test_labels = __create_sequences(test_df.values, seq_length)

    # Convert sequences to tensors
    train_sequences = torch.tensor(train_sequences, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_sequences = torch.tensor(test_sequences, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    # Getting loaders and epochs
    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader