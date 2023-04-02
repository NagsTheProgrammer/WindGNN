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
        'Wind Dir. 10 m Avg. (°)'
    ]

    return df[relevant_features].to_numpy()
