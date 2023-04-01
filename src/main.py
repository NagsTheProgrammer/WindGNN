from step1_loading_preprocessing import *
from step2_graph_builder import *
from step3_feature_extractor import *
from step4_gcn_layer_model import *
from step6_sequence_preparer import *
from step5_gcn_gru_combined_model import *
# from step8_trainer import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

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
    predictions = []
    truth = []
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
            predictions.append(test_out_inv_norm)
            truth.append(test_lab_inv_norm)

    # Step 7 - Printing Results
    stats1 = []
    stats2 = []
    stats3 = []

    for i in range(7):
        one_hour_prediction = [l[-1, i] for l in predictions]
        one_hour_truth = [l[0,-1, i] for l in truth]
        rms1 = mean_squared_error(one_hour_truth, one_hour_prediction, squared=False)

        one_hour_error = [l[-1, i] for l in test_loss_list]
        mae1 = np.average(np.array(one_hour_error))

        acc1 = np.array(one_hour_error)/np.array(one_hour_truth)
        acc_avg1 = np.average(acc1)
        acc_std1 = np.std(acc1)

        stats1.append([rms1, mae1, acc_avg1, acc_std1])
        ###
        two_hour_prediction = [l[-1, i+7] for l in predictions]
        two_hour_truth = [l[0,-1, i+7] for l in truth]
        rms2 = mean_squared_error(two_hour_truth, two_hour_prediction, squared=False)

        two_hour_error = [l[-1, i+7] for l in test_loss_list]
        mae2 = np.average(np.array(two_hour_error))

        acc2 = np.array(two_hour_error)/np.array(two_hour_truth)
        acc_avg2 = np.average(acc2)
        acc_std2 = np.std(acc2)

        stats2.append([rms2, mae2, acc_avg2, acc_std2])
        ###
        three_hour_prediction =[l[-1, i+14] for l in predictions]
        three_hour_truth = [l[0,-1, i+14] for l in truth]
        rms3 = mean_squared_error(three_hour_truth, three_hour_prediction, squared=False)

        three_hour_error = [l[-1, i+14] for l in test_loss_list]
        mae3 = np.average(np.array(three_hour_error))

        acc3 = np.array(three_hour_error)/np.array(three_hour_truth)
        acc_avg3 = np.average(acc3)
        acc_std3 = np.std(acc3)

        stats3.append([rms3, mae3, acc_avg3, acc_std3])
    
    col_labels = ['RMSE','MAE','Average Accuracy','Accuracy Deviation']
    one_hour_stats_df = pd.DataFrame(stats1, columns = col_labels, index = stations)
    two_hour_stats_df = pd.DataFrame(stats2, columns = col_labels, index = stations)
    three_hour_stats_df = pd. DataFrame(stats3, columns = col_labels, index = stations)

    print(one_hour_stats_df)
    print(two_hour_stats_df)
    print(three_hour_stats_df)
    
    one_hour = [l[-1, 0:7] for l in test_loss_list]
    two_hour = [l[-1, 7:14] for l in test_loss_list]
    three_hour = [l[-1, 14:21] for l in test_loss_list]

    one_hr_df = pd.DataFrame(one_hour, columns = stations)
    two_hr_df = pd.DataFrame(two_hour, columns = stations)
    thr_hr_df = pd.DataFrame(three_hour, columns = stations)

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

    one_hour_stats_df.to_csv('one_hour.csv', index = True)
    two_hour_stats_df.to_csv('two_hour.csv', index = True)
    three_hour_stats_df.to_csv('three_hour.csv', index = True)

    stuff = 0 # put breakpoint here if you want to examine the data
    #  ***