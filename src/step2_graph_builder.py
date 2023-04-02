import math

import numpy as np
from scipy.linalg import fractional_matrix_power


# Method to convert spherical latitude and longitude into planar mercator coordinates
def __convert_wgs2utm(coords):
    radius = 6378137
    for i in range(len(coords)):
        coords[i][0] = radius * math.log(math.tan(math.pi / 4 + coords[i][0] * math.pi / 360))
        coords[i][1] = radius * (coords[i][1] * math.pi / 180)
    return coords


def build_graph(df):
    # Extract unique station IDs and their coordinates
    stations = df[["Station Name", "Latitude", "Longitude"]].drop_duplicates()
    station_coordinates = stations[["Latitude", "Longitude"]].values
    station_coordinates = __convert_wgs2utm(station_coordinates)

    # Create a self looped adjacency matrix from the data
    # A = np.array(nx.attr_matrix(graph)[0])
    # A = nx.to_numpy_matrix(graph)
    dLength = len(station_coordinates)
    A_hat = np.ones((dLength, dLength))
    for i in range(dLength):
        for j in range(dLength):
            if i != j:
                X = station_coordinates[i][0] - station_coordinates[j][0]
                Y = station_coordinates[i][1] - station_coordinates[j][1]
                H = (X * X + Y * Y) / 100000000
                A_hat[i][j] = 1 / (math.sqrt(H))

    # Create a self-looped adjacency matrix from A
    # A_hat = A + np.identity(len(A))

    # Create a degree matrix from A_hat
    D = np.diag(np.sum(A_hat, axis=0))

    # Create a propagated adjacency matrix using A_hat and D
    D_half_norm = fractional_matrix_power(D, -0.5)
    A_star = D_half_norm.dot(A_hat).dot(D_half_norm)

    return A_star
