# script/proxy_target_engineering.py

import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def generate_proxy_target(df, snapshot_date=None, n_clusters=3, random_state=42):
    """
    Generates a binary target 'is_high_risk' using RFM and KMeans clustering.

    Parameters:
        df (pd.DataFrame): The raw transaction data.
        snapshot_date (str or None): The reference date for calculating recency.
        n_clusters (int): Number of clusters to form with KMeans.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame with CustomerId and is_high_risk column.
    """

    # Ensure datetime format
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Define snapshot date (1 day after the max date in data if not given)
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    # Step 1: RFM calculation per customer
    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,  # Recency
        "CustomerId": "count",  # Frequency
        "Amount": "sum"  # Monetary
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm.reset_index(inplace=True)

    # Step 2: Scale features
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # Step 3: KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm["Cluster"] = kmeans.fit_predict(scaled_rfm)

    # Step 4: Identify high-risk cluster
    # Usually: high Recency, low Frequency, low Monetary
    cluster_profiles = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_profiles.sort_values(by=["Recency", "Frequency", "Monetary"], ascending=[False, True, True]).index[0]

    # Step 5: Create binary target
    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]
