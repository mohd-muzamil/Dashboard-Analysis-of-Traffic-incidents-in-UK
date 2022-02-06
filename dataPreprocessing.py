import pandas as pd
from pandas_profiling import ProfileReport
import time
import math
from readData import *


def read_accidents_data(file):
    # Accidents Dataset
    accidents_data = pd.read_csv('Data/{}.csv'.format(file), error_bad_lines=False, warn_bad_lines=False,
                                 low_memory=False)

    # Keeping only those columns which are useful for Visualization
    accidents_data_useful_columns = [
        'Accident_Index',
        'Longitude',
        'Latitude',
        'Police_Force',
        'Accident_Severity',
        'Number_of_Vehicles',
        'Number_of_Casualties',
        'Date',
        'Day_of_Week',
        'Time',
        'Road_Type',
        'Speed_limit',
        'Light_Conditions',
        'Weather_Conditions',
        'Road_Surface_Conditions',
        'Urban_or_Rural_Area'
    ]

    accidents_data = accidents_data[accidents_data_useful_columns]

    accidents_data['Date'] = pd.to_datetime(accidents_data['Date'], format='%Y-%m-%d')

    # Removing rows which have no data represented by values=-1
    for col in accidents_data.columns:
        accidents_data = (accidents_data[accidents_data[col] != -1])

    # Data Imputing
    accidents_data = accidents_data[~accidents_data['Date'].isna()]
    accidents_data = accidents_data[~accidents_data['Time'].isna()]
    accidents_data = accidents_data[accidents_data['Number_of_Casualties'].astype(int) < 9]
    accidents_data = accidents_data[accidents_data['Number_of_Vehicles'].astype(int) < 9]
    accidents_data = accidents_data[~accidents_data['Longitude'].isna()]
    accidents_data = accidents_data[~accidents_data['Latitude'].isna()]
    accidents_data['Weather_Conditions'].fillna(value='Unknown')
    accidents_data['Road_Surface_Conditions'].fillna(value='Dry')

    accidents_data['Date'] = pd.to_datetime(accidents_data['Date'], dayfirst=True)
    accidents_data['year'] = pd.DatetimeIndex(accidents_data['Date']).year
    accidents_data['month'] = pd.DatetimeIndex(accidents_data['Date']).month
    accidents_data['hour'] = pd.DatetimeIndex(accidents_data['Time']).hour
    accidents_data['day'] = pd.DatetimeIndex(accidents_data['Date']).day
    accidents_data['week_in_month'] = pd.to_numeric(accidents_data.day / 7)
    accidents_data['week_in_month'] = accidents_data['week_in_month'].apply(lambda x: math.ceil(x))
    
    return accidents_data


def read_vehicles_data(file):
    # Vehicles Dataset
    vehicles_data = pd.read_csv('Data/{}.csv'.format(file), error_bad_lines=False, warn_bad_lines=False,
                                low_memory=False)

    # Keeping only those columns which are useful for Visualization
    vehicles_data_important_columns = [
        'Accident_Index',
        'Vehicle_Reference',
        'Vehicle_Type',
        'Sex_of_Driver',
        'Age_of_Driver',
        'Age_Band_of_Driver',
        'Engine_Capacity_(CC)',
        'Age_of_Vehicle'
    ]
    vehicles_data = vehicles_data[vehicles_data_important_columns]

    # Data Imputing, replacing -1 and NAN values with mean values
    for col in vehicles_data.columns[1:]:
        vehicles_data[col].fillna(int(round(vehicles_data[col].mean())), inplace=True)
        vehicles_data.loc[vehicles_data[col] == -1, col] = int(round(vehicles_data[col].mean()))

    return vehicles_data


if __name__ == "__main__":
    start_time = time.time()
    accidents_file_name = "Accident_Information"
    vehicles_file_name = "Vehicles0515"
    print('Start')
    df1 = read_accidents_data(accidents_file_name)
    df1.to_csv(r'Data/%s_out.csv' % accidents_file_name)
    profile = ProfileReport(df1, title="Pandas Profiling Report 1")
    profile.to_file('accidents_data_ProfileReport.html')
    print('accidents_data pre-processing done')

    df2 = read_vehicles_data(vehicles_file_name)
    df2.to_csv(r'Data/%s_out.csv' % vehicles_file_name)
    profile = ProfileReport(df2, title="Pandas Profiling Report 2")
    profile.to_file('vehicles_data_ProfileReport.html')
    print('vehicles_data pre-processing done')

    end_time = time.time()
    print('Total Time = ', (start_time - end_time) / 60, 'mins')
