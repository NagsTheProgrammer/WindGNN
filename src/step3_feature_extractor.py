import pandas as pd

def extract_features(df):
    # Selects the relevant features and returns them in a numpy format
    relevant_features = [
        "Station Name",
        "Date (Local Standard Time)",
        'Air Temp. Inst. (°C)',
        'Est. Dew Point Temp. (°C)',
        'Air Temp. Min. (°C)',
        'Air Temp. Max. (°C)',
        'Air Temp. Avg. (°C)',
        'Humidity Inst. (%)',
        'Relative Humidity Avg. (%)',
        'Precip. Accumulated (mm)',
        'Precip. (mm)',
        'Wind Speed 10 m Syno. (km/h)',
        'Wind Dir. 10 m Syno. (°)',
        'Wind Speed 10 m Avg. (km/h)',
        'Wind Dir. 10 m Avg. (°)',
        'Wind Dir EW Syno',
        'Wind Dir EW Avg'
    ]

    relevant_features = [
        "Station Name",
        "Date (Local Standard Time)",
        'Air Temp. Inst. (°C)',
        'Est. Dew Point Temp. (°C)',
        'Air Temp. Min. (°C)',
        'Air Temp. Max. (°C)',
        'Air Temp. Avg. (°C)',
        'Humidity Inst. (%)',
        'Relative Humidity Avg. (%)',
        'Precip. Accumulated (mm)',
        'Precip. (mm)',
        'Wind Speed 10 m Syno. (km/h)',
        'Wind Dir. 10 m Syno. (°)',
        'Wind Speed 10 m Avg. (km/h)',
        'Wind Dir. 10 m Avg. (°)',
        'Wind Dir EW Syno',
        'Wind Dir EW Avg'
    ]

    # relevant_features = [
    #     "Station Name",
    #     "Date (Local Standard Time)",
    #     'Wind Speed 10 m Syno. (km/h)',
    #     'Wind Speed 10 m Avg. (km/h)'
    # ]

    # relevant_features = [
    #     "Station Name",
    #     "Date (Local Standard Time)",
    #     'Wind Speed 10 m Syno. (km/h)',
    #     'Wind Dir. 10 m Syno. (°)',
    #     'Wind Speed 10 m Avg. (km/h)',
    #     'Wind Dir. 10 m Avg. (°)'
    # ]

    n = 26301
    m = 34
    # list_df = [attr_matrix[i:i+n] for i in range(0,attr_matrix.shape[0],n)]
    # print([i for i in list_df])
    # print(list_df[0].iloc[0])
    # attr_frames = [pd.concat(i.iloc[0]) for i in list_df]
    # t = attr_matrix[0:1]
    # print(t)
    attr_matrix = df[relevant_features]
    attr_frames = []
    for j in range(n):
        attr_rows = [attr_matrix[i+j:i+j+1] for i in range(0,attr_matrix.shape[0],n)]
        attr_frame = pd.concat(attr_rows)
        attr_frame = attr_frame.drop("Date (Local Standard Time)", axis=1)
        attr_frame = attr_frame.reset_index()
        attr_frame = attr_frame.drop("index", axis=1)
        attr_frame = attr_frame / attr_frame.values.max(axis=1).reshape(-1, 1)
        attr_frames.append(attr_frame)
        # print(j)

    return attr_matrix, attr_frames
