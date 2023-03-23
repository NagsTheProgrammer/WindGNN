import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def __create_sequences(data, seq_length):
    xs, ys = [], []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length), :]
        y = data[(i + seq_length), :]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def generate_sequences(df, model):
    # Get the output of the GCN
    gcn_output = model(data).detach().numpy()

    # Prepare the data for sequence generation
    df = df.set_index("Date/Time")
    df = df.loc[:, gcn_output.columns]
    df["Wind Speed 10 m Avg. (km/h)"] = gcn_output.squeeze()

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Create sequences
    seq_length = 24  # 24-hour sequences
    train_sequences, train_labels = __create_sequences(train_df.values, seq_length)
    test_sequences, test_labels = __create_sequences(test_df.values, seq_length)

    # Convert sequences to tensors
    train_sequences = torch.tensor(train_sequences, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_sequences = torch.tensor(test_sequences, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)
