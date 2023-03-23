from step1_loading_preprocessing import *
from step2_graph_builder import *
from step3_feature_extractor import *
from step4_gcn_model import *
from step5_sequence_preparer import *
from step6_gru_model import *
from step7_gcn_gru_combined_model import *
from step8_trainer import *

if __name__ == "__main__":
    # Step 1a - Load data from both csv (measurements and coordinates)
    # Step 1b - Preprocess data
    df = load_and_process_wind_speed_dataset()

    # Step 2 - Build distance graph
    graph = build_graph(df)

    # Step 3 - Select important features
    features_df = extract_features(df)

    input_dim = 111 # input size
    hidden_dim = 111 # hidden layer size
    layer_dim = 111 # how many layers
    output_dim = 111 # output size
    num_features = 111 # number of features from dataset
    hidden_channels = 111 # number of hidden channels
    num_classes = 111 # number of output classes ??? regression

    # model = ???
    model_GCN = GCNModel(input_dim, hidden_dim, layer_dim, output_dim)
    model_GRU = GRUModel(num_features, hidden_channels, num_classes)
    model_GCN_GRU = GCNGRUModel(model_GCN, model_GRU)

    if torch.cuda.is_available():
        model_GCN_GRU.cuda()

    criterion = nn.CrossEntropyLoss()