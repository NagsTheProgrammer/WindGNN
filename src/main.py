from step1_loading_preprocessing import *
from step2_graph_builder import *
from step3_feature_extractor import *
from step4_gcn_layer_model import *
from step6_sequence_preparer import *
from step5_gcn_gru_combined_model import *
# from step8_trainer import *
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Path to save best model to
    PATH = "./wind_gnn.pth"
    
    # Using GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Step 1a - Load data from both csv (measurements and coordinates)
    # Step 1b - Preprocess data
    df, wind_min, wind_max = load_and_process_wind_speed_dataset()

    # Step 2 - Build distance graph
    adj_matrix = build_graph(df)
    adj_matrix = torch.tensor(adj_matrix).float()
    adj_matrix = adj_matrix.to(device)

    # Step 3 - Select important features
    attr_matrix = extract_features(df)

    # Step 6 - generate train / test data sequences
    stations = np.unique(attr_matrix[:, 0])
    batch_size = 168
    train_loader, test_loader, num_attr, num_stations = generate_sequences(attr_matrix, batch_size, device)

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
    attr_station_flat = num_attr * num_stations
    num_predictions = num_stations * 3
    model = GCN_GRU(input_dim=num_attr, hidden_dim=num_attr, output_dim=num_attr, gru_input=attr_station_flat, gru_hidden_dim=num_predictions)
    model = model.to(device)
    
    

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
    test_loss_list = []
    iter = 0
    
    best_loss = 0.03
    patience = 0
    
    for epoch in range(epochs): # Repeating for every epoch
        for i, (batch_x, batch_y) in enumerate(train_loader): # for each batch in the train_loader
            outputs = model(adj_matrix, batch_x)
            
            # clear the gradients
            optimizer.zero_grad()

            # loss
            loss = lossFunction(outputs, batch_y)
            out_inv_norm = outputs.detach().cpu().numpy() * (wind_max - wind_min) + wind_min
            lab_inv_norm = batch_y.detach().cpu().numpy() * (wind_max - wind_min) + wind_min
            diff = abs(lab_inv_norm[0] - out_inv_norm)
            loss_list.append(diff)
            
            # backpropagation
            loss.backward()
            optimizer.step()
            iter += 1

            if loss.item() < best_loss:
                torch.save(model.state_dict(), PATH)
                best_loss = loss.item()
                patience = 0


            if iter % 100 == 0:
                print("epoch: %d, iter: %d, patience: %d, loss: %1.5f" % (epoch, iter, patience, loss.item()))
        iter = 0
        patience += 1
        if patience > 10:
            break
    
    model = GCN_GRU(input_dim=num_attr, hidden_dim=num_attr, output_dim=num_attr, gru_input=attr_station_flat, gru_hidden_dim=num_predictions)
    model = model.to(device)
    model.load_state_dict(torch.load(PATH))
    with torch.no_grad():
        for (batch_x, batch_y) in test_loader:
            outputs = model(adj_matrix, batch_x)
            test_out_inv_norm = outputs.detach().cpu().numpy() * (wind_max - wind_min) + wind_min
            test_lab_inv_norm = batch_y.detach().cpu().numpy() * (wind_max - wind_min) + wind_min
            test_diff = abs(test_lab_inv_norm[0] - test_out_inv_norm)
            test_loss_list.append(test_diff)

    # Step 7 - Printing Results
    ll = loss_list
    tll = test_loss_list
    
    plot_data = []
    n = 0

    one_hour = [l[-1, 0:7] for l in tll]
    two_hour = [l[-1, 7:14] for l in tll]
    three_hour = [l[-1, 14:21] for l in tll]

    one_hr_df = pd.DataFrame(one_hour, columns = stations)
    two_hr_df = pd.DataFrame(two_hour, columns = stations)
    thr_hr_df = pd.DataFrame(three_hour, columns = stations)

    print(one_hr_df)

    fig1, ax1 = plt.subplots()
    ax1.boxplot(one_hr_df)
    plt.xticks([1,2,3,4,5,6,7], stations, rotation=45)
    plt.ylabel('Wind Speed (km/hr)')
    plt.xlabel('Weather Station')
    plt.title("Absolute Error for One-Hour Prediction")
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.boxplot(two_hr_df)
    plt.xticks([1,2,3,4,5,6,7], stations, rotation=45)
    plt.ylabel('Wind speed (km/hr)')
    plt.xlabel('Weather Station')
    plt.title("Absolute Error for Two-Hour Prediction")
    plt.show()

    fig3, ax3 = plt.subplots()
    ax3.boxplot(thr_hr_df)
    plt.xticks([1,2,3,4,5,6,7], stations, rotation=45)
    plt.ylabel('Wind speed (km/hr)')
    plt.xlabel('Weather Station')
    plt.title("Absolute Error for Three-Hour Prediction")
    plt.show()


    stuff = 0 # put breakpoint here if you want to examine the data
    #  ***