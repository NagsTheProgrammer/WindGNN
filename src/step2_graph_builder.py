from scipy.spatial import Delaunay
import networkx as nx
import math


def convert_wgs2utm(coords):
    radius = 6378137
    for i in range(len(coords)):
        coords[i][0] = radius * math.log(math.tan(math.pi / 4 + coords[i][0] * math.pi / 360))
        coords[i][1] = radius * (coords[i][1] * math.pi / 180)
    return coords


def build_graph(df):
    # Extract unique station IDs and their coordinates
    stations = df[["Station ID", "Latitude", "Longitude"]].drop_duplicates()
    station_coordinates = stations[["Latitude", "Longitude"]].values
    station_coordinates = convert_wgs2utm(station_coordinates)

    # Perform Delaunay triangulation
    tri = Delaunay(station_coordinates)

    # Create a NetworkX graph from the Delaunay triangulation
    graph = nx.Graph()
    graph.add_nodes_from(stations["Station ID"].values)

    # Iterate over the Delaunay triangles and add edges to the graph
    for triangle in tri.simplices:
        a, b, c = triangle
        graph.add_edges_from([(stations.iloc[a]["Station ID"], stations.iloc[b]["Station ID"]),
                              (stations.iloc[b]["Station ID"], stations.iloc[c]["Station ID"]),
                              (stations.iloc[c]["Station ID"], stations.iloc[a]["Station ID"])])

    return graph


# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pathlib import Path
# from scipy.linalg import fractional_matrix_power
#
# # Function to split the data
# def split(df):
#     return np.array_split(df, 7)
#
# # Initialize graph
# # G = nx.Graph(name='Weather Station Graph')
#
# # Import data
# filePath = Path(r'C:\Users\hxi-c\OneDrive\Documents\GitHub\WindGNN\data\ACISHourlyData-20200101-20200630-PID181534737.csv')
# data = pd.read_csv(filePath, encoding='cp1252')
#
# # Split data into nodes
# nodes = split(data)
#
# # Add nodes to graph
# # for i in range(len(nodes)):
#     # G.add_node(i, name=nodes[i].iloc[0][0], time=nodes[i].iloc[0][1], inst_temp=nodes[i].iloc[0][2], dewPoint_temp=nodes[i].iloc[0][5], min_temp=nodes[i].iloc[0][7],
#     #            max_temp=nodes[i].iloc[0][10], avg_temp=nodes[i].iloc[0][13], inst_humid=nodes[i].iloc[0][16], avg_rel_humid=nodes[i].iloc[0][19], acc_precip=nodes[i].iloc[0][22],
#     #            precip=nodes[i].iloc[0][25], acc_precip_wg=nodes[i].iloc[0][28], precip_wg=nodes[i].iloc[0][31], wind_speed=nodes[i].iloc[0][34], wind_dir=nodes[i].iloc[0][37],
#     #            wind_speed_avg=nodes[i].iloc[0][40], wind_dir_avg=nodes[i].iloc[0][43])
#     # G.add_node(i, name=nodes[i].iloc[0][0])
#
# # Add edges to graph
# # for u in G.nodes:
# #     for v in G.nodes:
# #         if u != v:
# #             G.add_edge(u,v)
#
# # Making sure graph is created with proper information
# # print('Graph info:\n', nx.info(G))
# # print('\nGraph nodes: ', G.nodes.data())
# # nx.draw(G, with_labels=True, font_weight='bold')
# # plt.show()
#
# # Get the adjacency matrix (A) and node feature matrix (X)
# # A = np.array(nx.attr_matrix(G)[0])
# # X = np.array(nx.attr_matrix(G, node_attr='wind_speed_avg')[1])
# # X = np.expand_dims(X, axis=1)
#
# # This is the weighted adjacency matrix (A) for the graph. Weights are calculated from the GPS coordinates of each station
# # I multiplied each coordinate by 10, and then calculated the magnitude between each point. The weight is the inverse of this magnitude
# # WILL WORK ON IMPROVING THIS
# A = np.array([[0.000 , 0.324 , 0.293 , 0.118 , 0.127 , 0.246 , 0.198],
#               [0.324 , 0.000 , 0.204 , 0.179 , 0.255 , 0.467 , 0.494],
#               [0.293 , 0.204 , 0.000 , 0.147 , 0.127 , 0.302 , 0.224],
#               [0.118 , 0.179 , 0.147 , 0.000 , 0.253 , 0.225 , 0.273],
#               [0.127 , 0.255 , 0.127 , 0.253 , 0.000 , 0.219 , 0.294],
#               [0.246 , 0.467 , 0.302 , 0.225 , 0.219 , 0.000 , 0.837],
#               [0.198 , 0.494 , 0.224 , 0.273 ,0.294 ,  0.837 , 0.000]])
#
# # To make a self-looped adjacency matrix (A_hat), add a diagonal of ones
# A_hat = A + np.identity(7)
#
# # The feature matrix (X). For simplicity this is only the wind speed
# X = np.array([[nodes[0].iloc[0][40]],
#               [nodes[1].iloc[0][40]],
#               [nodes[2].iloc[0][40]],
#               [nodes[3].iloc[0][40]],
#               [nodes[4].iloc[0][40]],
#               [nodes[5].iloc[0][40]],
#               [nodes[6].iloc[0][40]]])
#
# # The degree matrix (D) tells us how much weight is on a node from the edges it is connected to
# D = np.diag(np.sum(A_hat, axis=0))
#
# # To noromalize our self-looped adjacency matrix (A_hat) as Kipf and Welling tell us, we take the
# # inverse *square root* of the degree matrix and matrix mulitply to get (A_star)
# D_half_norm = fractional_matrix_power(D, 0.5)
# A_star = D_half_norm.dot(A_hat).dot(D_half_norm)
#
# # Multiply (A_star) and (X) to get our normalized features (H)
# H = A_star.dot(X)
#
# # Multiply (H) by our weight (W) and pass the product into our activation function (ReLU) to get (H2)
# # Our weight (W) is the Chebyshev polynomial. WILL WORK ON IMPLEMENTING THIS
# # Multpily (A_star)*(H2)*(W) and pass this product into our sigma function to get our learned spatial features to pass into our GRU model