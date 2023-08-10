import pandas as pd
import torch


def load_and_process_wind_speed_dataset(verbose: bool = True, dataset_size: bool = True):
    # read in the csv file
    coordinates_df = pd.read_csv(r'data/ACISStationCoordinates.csv')
    if dataset_size:
        full_df = pd.read_csv(r'data/ACISHourlyData-20200101-20221231_large.csv')
    else:
        full_df = pd.read_csv(r'data/ACISHourlyData-20200101-20221231.csv')
    

    # remove useless fields
    full_df = full_df.drop(columns=["Unnamed: 0",
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

    # remove unwanted stations
    full_df = full_df[full_df['Station Name'] != 'Enchant 2 AGCM']

    # Change direction columns
    full_df['Wind Dir. 10 m Syno. (°)'] = full_df['Wind Dir. 10 m Syno. (°)'] - 180
    full_df['Wind Dir. 10 m Avg. (°)'] = full_df['Wind Dir. 10 m Avg. (°)'] - 180
    full_df['Wind Dir EW Syno'] = abs(full_df['Wind Dir. 10 m Syno. (°)']) - 90
    full_df['Wind Dir EW Avg'] = abs(full_df['Wind Dir. 10 m Avg. (°)']) - 90

    # select numerical features
    numerical_features = ['Air Temp. Inst. (°C)', 'Est. Dew Point Temp. (°C)', 'Air Temp. Min. (°C)',
                          'Air Temp. Max. (°C)', 'Air Temp. Avg. (°C)', 'Humidity Inst. (%)',
                          'Relative Humidity Avg. (%)', 'Precip. Accumulated (mm)', 'Precip. (mm)',
                          'Precip. Accumulated (WG) (mm)', 'Precip. (WG) (mm)', 'Wind Speed 10 m Syno. (km/h)',
                          'Wind Dir. 10 m Syno. (°)', 'Wind Speed 10 m Avg. (km/h)', 'Wind Dir. 10 m Avg. (°)',
                          'Sclerotinia Infection Risk', 'Wind Dir EW Syno', 'Wind Dir EW Avg']

    # add nullable flags
    for col in full_df.columns:
        if full_df[col].isnull().sum() > 0:
            full_df[f'{col}_present'] = full_df[col].notnull()

    # fill in missing values
    for col in numerical_features:
        full_df[col].fillna(full_df[col].mean(), inplace=True)
    full_df.fillna("0", inplace=True)

    # add coordinates
    full_df = full_df.merge(coordinates_df, on='Station Name')

    # normalize data, retain min and max for wind speed
    wind_data = torch.tensor(full_df['Wind Speed 10 m Avg. (km/h)'].values)
    wind_data_min = torch.min(wind_data, dim=0, keepdim=True)[0].numpy()
    wind_data_max = torch.max(wind_data, dim=0, keepdim=True)[0].numpy()

    numerical_data = torch.tensor(full_df[numerical_features].values)
    numerical_data_min = torch.min(numerical_data, dim=0, keepdim=True)[0]
    numerical_data_max = torch.max(numerical_data, dim=0, keepdim=True)[0]
    normalized_numerical_data = (numerical_data - numerical_data_min) / (numerical_data_max - numerical_data_min) # from 0 to 1
    # normalized_numerical_data = (numerical_data - (numerical_data_min + numerical_data_max)/2) / (numerical_data_max - numerical_data_min) / 2 # from -1 to 1

    # normalized_numerical_data[:,11] = (normalized_numerical_data[:,11] - torch.min(normalized_numerical_data[:,11])) / (torch.max(normalized_numerical_data[:,11]) - torch.min(normalized_numerical_data[:,11]))
    # normalized_numerical_data[:,13] = (normalized_numerical_data[:,13] - torch.min(normalized_numerical_data[:,13])) / (torch.max(normalized_numerical_data[:,13]) - torch.min(normalized_numerical_data[:,13]))
    
    normalized_numerical_data[:,12] = ( normalized_numerical_data[:,12] - (torch.min(normalized_numerical_data[:,12]) + torch.max(normalized_numerical_data[:,12])) / 2 ) / (torch.max(normalized_numerical_data[:,12]) - torch.min(normalized_numerical_data[:,12])) / 2
    normalized_numerical_data[:,14] = ( normalized_numerical_data[:,14] - (torch.min(normalized_numerical_data[:,14]) + torch.max(normalized_numerical_data[:,14])) / 2 ) / (torch.max(normalized_numerical_data[:,14]) - torch.min(normalized_numerical_data[:,14])) / 2
    normalized_numerical_data[:,16] = ( normalized_numerical_data[:,16] - (torch.min(normalized_numerical_data[:,16]) + torch.max(normalized_numerical_data[:,16])) / 2 ) / (torch.max(normalized_numerical_data[:,16]) - torch.min(normalized_numerical_data[:,16])) / 2
    normalized_numerical_data[:,17] = ( normalized_numerical_data[:,17] - (torch.min(normalized_numerical_data[:,17]) + torch.max(normalized_numerical_data[:,17])) / 2 ) / (torch.max(normalized_numerical_data[:,17]) - torch.min(normalized_numerical_data[:,17])) / 2

    test = normalized_numerical_data.numpy()
    full_df[numerical_features] = normalized_numerical_data.numpy()
    speed1 = normalized_numerical_data.select(1,11)
    speed2 = normalized_numerical_data[:][13]

    if verbose:
        print("Loaded and preprocessed data. Shape:", full_df.shape)

    return full_df, wind_data_min, wind_data_max
