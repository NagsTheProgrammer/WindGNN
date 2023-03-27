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

    model = GCN_GRU(input_dim=13, hidden_dim=13, output_dim=13, gru_hidden_dim=13)

    # ---- In progress
    # attr_matrix = np.array_split(df, 7)  # splitting df into each station *** works for small dataset, may not work for larger dataset as total timesteps for each station is unbalanced
    # dLength = len(attr_matrix[0])

    attr_matrix_transpose = pd.DataFrame()
    station_specific_df_transpose = pd.DataFrame()
    station_names = attr_matrix['Station Name'].unique()

    for station in station_names:
        station_specific_df = attr_matrix[(attr_matrix['Station Name'] == station)]
        station_specific_df_transpose = station_specific_df.transpose()
        attr_matrix_transpose = attr_matrix_transpose.append(station_specific_df_transpose)
        # (station, feature, timestep) = [7, 15, 26,000]
        # attr_matrix_transpose.append(attr_matrix[(attr_matrix['Station Name'] == station)].transpose())  # finds matrix subset with single station, transposes, and appends to attr_matrix_transpose

    # if above didn't work, this should
    # attr_matrix_transpose = []
    # for a in attr_matrix:
    #     attr_matrix_transpose = np.append(attr_matrix_transpose, np.transpose(a)) # transposing into [feature, timestep]

    attr_matrix_stack = np.stack(attr_matrix_transpose)

    print(attr_matrix_stack.shape)  # printing shape - expected (7, 13, 184107) = [station, faeture, timestep]

    print("Done")
