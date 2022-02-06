from dataPreprocessing import *

    
def read_preprocessed_accidents_data():
    accident_data = pd.read_csv('Data/Accident_Information_out.csv', low_memory=False)
    return accident_data


def read_preprocessed_vehicles_data():
    vehicles_data = pd.read_csv('Data/Vehicle_Information_out.csv', low_memory=False)
    return vehicles_data


def readAccidentVehiclesData_before_merge():
    accident_data = read_preprocessed_accidents_data()
    vehicle_data = read_preprocessed_vehicles_data()
    df = accident_data.merge(vehicle_data, how='left', on='Accident_Index')
    return df


def readAccidentVehiclesData():
    combined_data = pd.read_csv('Data/combined_data.csv', low_memory=False)
    return combined_data


if __name__ == "__main__":
    accident_data = read_accidents_data("Accident_Information")
    accident_data.to_csv('Data/Accident_Information_out.csv', index=False)
    print("Accident data processed")

    vehicles_data = read_vehicles_data("Vehicles0515")
    vehicles_data.to_csv('Data/Vehicle_Information_out.csv', index=False)
    print("Vehicle data processed")

    df = readAccidentVehiclesData_before_merge()
    df.to_csv('Data/combined_data.csv', index=False)
