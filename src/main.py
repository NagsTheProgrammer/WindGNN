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
    df, data_min, data_max = load_and_process_wind_speed_dataset()

    # Step 2 - Build distance graph
    adj_matrix = build_graph(df)
    adj_matrix = torch.tensor(adj_matrix).float()
    dLength = len(adj_matrix)

    # Step 3 - Select important features
    attr_matrix = extract_features(df)
    # attr_matrix = np.array_split(extract_features(df), dLength)

    # Step 6 - generate train / test data sequences
    batch_size = 168
    train_loader, test_loader = generate_sequences(attr_matrix, batch_size)

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
    model = GCN_GRU(input_dim=13, hidden_dim=13, output_dim=13, gru_input=91, gru_hidden_dim=7)

    # Using GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Step 5 - Generate Train / Test Data Sequences

    # n_iters = 100 # arbitrarily chosen, one cycle of
    # batch_size = 64 # arbitrarily chosen
    # train_loader, test_loader = generate_sequences(df, model, batch_size)

    # Step 6 - Train GRU

    # hyperparameters to explore:
    # dropout - [0.0, 1.1]
    # learning rate - scheduling(?) - [0.0001, 0.1]
    # epochs - early stopping(?) - [10, 300]
    # hidden state size -  [2, 256]
    # number of hidden layers - [1, 4]
    # batch size - [4, 8, 16, 32, 64, 128, 256]
    # sequence length - [12, 24, 48, 168]

    learning_rate = 0.001
    epochs = 100 # number of times the model sees the complete dataset

    # *** From previous iteration where GRU and GCN were separate models
    # gru_model = GRUModel(input_size, hidden_size, num_layers, output_size)

    # Defining Loss Function
    # lossFunction = nn.L1loss() # Mean Absolute Error - used when data has significant outliers from mean value
    lossFunction = nn.MSELoss() # Mean Squared Error - default for regression problems

    # Defining optimizer
    # (params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    iter = 0

    for epoch in range(epochs): # Repeating for every epoch
        for i, (batch_x, batch_y) in enumerate(train_loader): # for each batch in the train_loader
            outputs = model(adj_matrix, batch_x)
            # sh = outputs.shape
            # o = outputs.detach().numpy()
            # clear the gradients
            optimizer.zero_grad()
            # loss
            loss = lossFunction(outputs, batch_y)
            # backpropagation
            loss.backward()
            optimizer.step()
            iter += 1
            if iter % 100 == 0:
                print("epoch: %d, iter: %d, loss: %1.5f" % (epoch, iter, loss.item()))
        iter = 0

    # Step 7 - Printing Results

    #  ***