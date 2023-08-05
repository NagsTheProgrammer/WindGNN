import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from step1_loading_preprocessing import *
from step2_graph_builder import *
from step3_feature_extractor import *
from step6_gcn_gru_combined_model import *
from step4_sequence_preparer import *
from step7_DQN import *

if __name__ == "__main__":
    
    # Path to save best model to
    PATH = "./wind_gnn_34.pth"

    # Using GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    test = [[[4,2,7,1],[2,4,7,16],[10,8,2,4]],[[11,12,130,14],[6,5,7,8],[101,6,7,8]],[[4,2,7,1],[2,4,7,16],[10,8,2,4]],[[11,12,130,14],[6,5,7,8],[101,6,7,8]]]
    test = torch.tensor(test)
    # t = test.max(2)[0]
    # print(t)
    # t = test.max(2)[1]
    # print(t)
    # t = test.max(2)[0].max(1)[0]
    # print(t)
    # t = test.max(2)[0].max(1)[1]
    # print(t)
    # t = test.max(2)[0].max(1)[0].max(0)[0]
    # print(t)
    # t = test.max(2)[0].max(1)[0].max(0)[1]
    # print(t)

    # b0 = test.max(2)[0].max(1)[0].max(0)[1]
    # b1 = test.max(2)[0].max(1)[1][b0]
    # b3 = test.max(2)[1][b0][b1]
    # t = test.max(2)[0][b0][b1]
    # print(t)
    # print(b3)


    # Step 1a - Load data from both csv (measurements and coordinates)
    # Step 1b - Preprocess data
    
    # Set large to True if you want to use the full dataset
    large = True
    df, wind_min, wind_max = load_and_process_wind_speed_dataset(dataset_size=large)

    # Step 2 - Build distance graph
    root_graph = build_A_hat(df)
    adj_matrix = build_graph(df)
    adj_matrix = torch.tensor(adj_matrix).float()
    adj_matrix = adj_matrix.to(device)

    # Step 3 - Select important features
    attr_matrix = extract_features(df)

    # Step 4 - generate train / test data sequences
    stations = np.unique(attr_matrix[:, 0])
    num_stations = len(stations)
    batch_size = 168
    train_loader, test_loader, num_attr, num_stations = generate_sequences(attr_matrix, batch_size, device)

    # Step 5 - Defining Two-Layer GCN with GRU
    attr_station_flat = num_attr * num_stations
    num_predictions = num_stations * 3
    model = GCN_GRU(input_dim=num_attr, hidden_dim=num_attr, output_dim=num_attr, gru_input=attr_station_flat,
                    gru_hidden_dim=num_predictions)
    model = model.to(device)

    # Create a list of actions for RL agent
    actions = createActions()

    learning_rate = 0.001
    epochs = 150  # number of times the model sees the complete dataset

    # Defining Loss Function
    lossFunction = nn.MSELoss()  # Mean Squared Error - default for regression problems

    # Defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    test_loss_list = []
    predictions = []
    truth = []
    iter = 0

    best_loss = 0.03
    best_loss_temp = 0.002

    worst_loss = 0.003
    iterloader = enumerate(train_loader)
    tempInt, (batch, label) = next(iterloader)

    patience = 0
    
    # GCN-GRU-RL training loop
    time0 = datetime.datetime.now().timestamp()
    for epoch in range(epochs):  # Repeating for every epoch
        for i, (batch_x, batch_y) in enumerate(train_loader):  # for each batch in the train_loader
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

            # take the batch with the worst performance
            if loss.item() > worst_loss:
                batch = batch_x
                label = batch_y
                worst_loss = loss.item()
            
            # save the model with the best performance and reset patience
            if loss.item() < best_loss:
                torch.save(model.state_dict(), PATH)
                best_loss = loss.item()
                patience = 0
            
            if iter % 100 == 0:
                print("epoch: %d, iter: %d, patience: %d, loss: %1.5f, best loss: %1.5f" % (epoch, iter, patience, loss.item(), best_loss))
        
        # reset the worst loss to be the best loss
        worst_loss = best_loss
        iter = 0
        patience += 1
        if patience > 20:
            break
        
        # if the model has improved beyond the threshold, start fine tuning with reinforcement learning
        if best_loss < best_loss_temp:
            best_loss_temp = best_loss
            RL_Agent = DQNTrainer(model, root_graph, wind_max, wind_min, actions)
            adj_matrix = RL_Agent.train(adj_matrix, batch, label)
            del RL_Agent


    time1 = datetime.datetime.now().timestamp()
    trainingTime = time1 - time0

    # RL loop
    #for i, (batch_x, batch_y) in enumerate(train_loader):
    #    print("test")

    # Testing loop
    model = GCN_GRU(input_dim=num_attr, hidden_dim=num_attr, output_dim=num_attr, gru_input=attr_station_flat,
                    gru_hidden_dim=num_predictions)
    model = model.to(device)
    model.load_state_dict(torch.load(PATH))
    with torch.no_grad():
        for (batch_x, batch_y) in test_loader:
            outputs = model(adj_matrix, batch_x)
            test_out_inv_norm = outputs.detach().cpu().numpy() * (wind_max - wind_min) + wind_min
            test_lab_inv_norm = batch_y.detach().cpu().numpy() * (wind_max - wind_min) + wind_min + [1]
            test_diff = abs(test_lab_inv_norm[0] - test_out_inv_norm)
            test_loss_list.append(test_diff)
            predictions.append(test_out_inv_norm)
            truth.append(test_lab_inv_norm)

    # Step 7 - Printing Results
    stats1 = []
    stats2 = []
    stats3 = []

    for i in range(num_stations):
        one_hour_prediction = [l[-1, i] for l in predictions]
        one_hour_truth = [l[0, -1, i] for l in truth]
        rms1 = mean_squared_error(one_hour_truth, one_hour_prediction, squared=False)

        one_hour_error = [l[-1, i] for l in test_loss_list]
        mae1 = np.average(np.array(one_hour_error))

        acc1 = 1 - np.array(one_hour_error) / np.array(one_hour_truth)
        acc_avg1 = np.average(acc1)
        acc_std1 = np.std(acc1)

        stats1.append([rms1, mae1, acc_avg1, acc_std1])
        ###
        ###
        ###
        two_hour_prediction = [l[-1, i + num_stations] for l in predictions]
        two_hour_truth = [l[0, -1, i + num_stations] for l in truth]
        rms2 = mean_squared_error(two_hour_truth, two_hour_prediction, squared=False)

        two_hour_error = [l[-1, i + num_stations] for l in test_loss_list]
        mae2 = np.average(np.array(two_hour_error))

        acc2 = 1 - np.array(two_hour_error) / np.array(two_hour_truth)
        acc_avg2 = np.average(acc2)
        acc_std2 = np.std(acc2)

        stats2.append([rms2, mae2, acc_avg2, acc_std2])
        ###
        ###
        ###
        three_hour_prediction = [l[-1, i + 2*num_stations] for l in predictions]
        three_hour_truth = [l[0, -1, i + 2*num_stations] for l in truth]
        rms3 = mean_squared_error(three_hour_truth, three_hour_prediction, squared=False)

        three_hour_error = [l[-1, i + 2*num_stations] for l in test_loss_list]
        mae3 = np.average(np.array(three_hour_error))

        acc3 = 1 - np.array(three_hour_error) / np.array(three_hour_truth)
        acc_avg3 = np.average(acc3)
        acc_std3 = np.std(acc3)

        stats3.append([rms3, mae3, acc_avg3, acc_std3])

    col_labels = ['RMSE', 'MAE', 'Average Accuracy', 'Accuracy Deviation']
    one_hour_stats_df = pd.DataFrame(stats1, columns=col_labels, index=stations)
    two_hour_stats_df = pd.DataFrame(stats2, columns=col_labels, index=stations)
    three_hour_stats_df = pd.DataFrame(stats3, columns=col_labels, index=stations)

    #print(one_hour_stats_df)
    #print(two_hour_stats_df)
    #print(three_hour_stats_df)
    
    print(trainingTime)

    one_hour = [l[-1, 0 : num_stations] for l in test_loss_list]
    two_hour = [l[-1, num_stations : 2*num_stations] for l in test_loss_list]
    three_hour = [l[-1, 2*num_stations : 3*num_stations] for l in test_loss_list]

    one_hr_df = pd.DataFrame(one_hour, columns=stations)
    two_hr_df = pd.DataFrame(two_hour, columns=stations)
    thr_hr_df = pd.DataFrame(three_hour, columns=stations)
    
    station_ticks = num_stations + 1
    ticks = list(range(1,station_ticks))

    fig1, ax1 = plt.subplots()
    ax1.boxplot(one_hr_df)
    plt.xticks(ticks, stations, rotation=90)
    plt.ylabel('Wind Speed (km/hr)')
    plt.xlabel('Weather Station')
    plt.title("Absolute Error for One-Hour Prediction")
    plt.subplots_adjust(left = 0.2, bottom = 0.3, right = 0.8, top = 0.9, wspace = 0.2, hspace = 0.2)
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.boxplot(two_hr_df)
    plt.xticks(ticks, stations, rotation=90)
    plt.ylabel('Wind speed (km/hr)')
    plt.xlabel('Weather Station')
    plt.title("Absolute Error for Two-Hour Prediction")
    plt.subplots_adjust(left = 0.2, bottom = 0.3, right = 0.8, top = 0.9, wspace = 0.2, hspace = 0.2)
    plt.show()

    fig3, ax3 = plt.subplots()
    ax3.boxplot(thr_hr_df)
    plt.xticks(ticks, stations, rotation=90)
    plt.ylabel('Wind speed (km/hr)')
    plt.xlabel('Weather Station')
    plt.title("Absolute Error for Three-Hour Prediction")
    plt.subplots_adjust(left = 0.2, bottom = 0.3, right = 0.8, top = 0.9, wspace = 0.2, hspace = 0.2)
    plt.show()

    one_hour_stats_df.to_csv('one_hour.csv', index=True)
    two_hour_stats_df.to_csv('two_hour.csv', index=True)
    three_hour_stats_df.to_csv('three_hour.csv', index=True)