import networkx as nx
from scipy.spatial import Delaunay
from torch_geometric.data import Data

def build_graph(unique_stations)->:
    # Construct graph
    # ... (Extract coordinates and create nodes as shown in the previous example)

    # Delaunay triangulation
    tri = Delaunay(unique_stations[['latitude', 'longitude']].values)
    edges = set()

    for simplex in tri.vertices:
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            edges.add(edge)

    # Add edges to the graph
    for i, j in edges:
        station_i = unique_stations.iloc[i]['station_id']
        station_j = unique_stations.iloc[j]['station_id']
        graph.add_edge(station_i, station_j)
