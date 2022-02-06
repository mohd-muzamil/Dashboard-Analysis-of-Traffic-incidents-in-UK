from sklearn.cluster import DBSCAN
import numpy as np


def clustering(df, distance, min_samples):
    kms_per_radian = 6371.0088
    eps = distance / (kms_per_radian * 1000)
    min_samples = min_samples

    df_dbc = df

    coords = df[["Latitude", "Longitude"]].values
    dbc = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    labels = dbc.labels_

    df_dbc['Cluster'] = labels

    return df_dbc
