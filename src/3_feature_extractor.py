def extract_features(df):
    # Select the relevant features
    relevant_features = [
        "Station ID",
        "Date/Time",
        "Wind Speed 10 m Avg. (km/h)",
        "Air Temp. Inst. (°C)",
        "Est. Dew Point Temp. (°C)",
        "Humidity Inst. (%)",
        "Relative Humidity Avg. (%)",
        "Precip. Accumulated (mm)",
        "Precip. (mm)",
        "Precip. Accumulated (WG) (mm)",
        "Precip. (WG) (mm)",
    ]

    features_df = df[relevant_features]

    # Pivot the DataFrame to have features as columns indexed by station ID and date/time
    return features_df.pivot_table(index=["Station ID", "Date/Time"]).reset_index()
