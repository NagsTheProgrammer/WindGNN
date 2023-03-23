from scipy.spatial import Delaunay
import networkx as nx

def build_graph(df):
    # Extract unique station IDs and their coordinates
    stations = df[["Station ID", "Latitude", "Longitude"]].drop_duplicates()
    station_coordinates = stations[["Latitude", "Longitude"]].values

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
