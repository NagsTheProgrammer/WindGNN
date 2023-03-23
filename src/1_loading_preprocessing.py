import pandas as pd
import torch


def load_and_process_wind_speed_dataset(file_path: str):
    # read in the csv file
    df = pd.read_csv(file_path)

    # remove useless fields
    df = df.drop(columns=["Unnamed: 0",
                          "Air Temp. Inst. Comment",
                          "Air Temp. Min. Comment",
                          "Air Temp. Max. Comment",
                          "Air Temp. Avg. Comment",
                          "Humidity Inst. Comment",
                          "Relative Humidity Avg. Comment",
                          "Precip. Accumulated Comment",
                          "Precip. Comment",
                          "Precip. Accumulated (WG) Comment",
                          "Precip. (WG) Comment",
                          "Wind Speed 10 m Syno. Comment",
                          "Wind Dir. 10 m Syno. Comment",
                          "Wind Speed 10 m Avg. Comment",
                          "Wind Dir. 10 m Avg. Comment",
                          "Sclerotinia Infection Risk Comment"])

    # select numerical features
    numerical_features = ['Air Temp. Inst. (°C)', 'Est. Dew Point Temp. (°C)', 'Air Temp. Min. (°C)',
                          'Air Temp. Max. (°C)', 'Air Temp. Avg. (°C)', 'Humidity Inst. (%)',
                          'Relative Humidity Avg. (%)', 'Precip. Accumulated (mm)', 'Precip. (mm)',
                          'Precip. Accumulated (WG) (mm)', 'Precip. (WG) (mm)', 'Wind Speed 10 m Syno. (km/h)',
                          'Wind Dir. 10 m Syno. (°)', 'Wind Speed 10 m Avg. (km/h)', 'Wind Dir. 10 m Avg. (°)',
                          'Sclerotinia Infection Risk']

    # add nullable flags
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[f'{col}_present'] = df[col].notnull()

    # fill in missing values
    for col in numerical_features:
        df[col].fillna(df[col].mean(), inplace=True)
    df.fillna("0", inplace=True)

    # normalize data
    numerical_data = torch.tensor(df[numerical_features].values)
    numerical_data_min = torch.min(numerical_data, dim=0, keepdim=True)[0]
    numerical_data_max = torch.max(numerical_data, dim=0, keepdim=True)[0]
    normalized_numerical_data = (numerical_data - numerical_data_min) / (numerical_data_max - numerical_data_min)
    df[numerical_features] = normalized_numerical_data.numpy()
