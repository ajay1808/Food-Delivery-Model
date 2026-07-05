"""
City-Scale ETA Prediction & Routing Engine (v1.0)

Spatiotemporal feature engineering + CatBoost regression pipeline for
predicting food delivery ETA. See README.md for the full write-up.
"""

import time

import h3
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# Logging helper

def log_step(message, start_time=None):
    """Prints a timestamped progress message. Pass start_time to also print elapsed duration."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    if start_time:
        duration = time.time() - start_time
        print(f"[{timestamp}] {message} (took {duration:.2f}s)")
    else:
        print(f"[{timestamp}] {message}")


# Distance functions

def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in km."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    return r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def manhattan_distance(lat1, lon1, lat2, lon2):
    """L1 (grid) distance between two points, in km."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = np.abs(lat2 - lat1)
    dlon = np.abs(lon2 - lon1)
    return r * (dlat + np.cos(lat1) * dlon)


# Data cleaning and feature engineering

def process_data(train_path, test_path):
    t0 = time.time()
    log_step("Reading raw CSV files...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['is_train'] = 1
    test_df['is_train'] = 0
    df = pd.concat([train_df, test_df], ignore_index=True)

    log_step("Cleaning strings and resolving missing values...")
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip()).replace('NaN', np.nan)

    if 'Time_taken(min)' in df.columns:
        df['Time_taken(min)'] = df['Time_taken(min)'].astype(str).str.extract(r'(\d+)').astype(float)

    for col in ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Vehicle_condition']:
        df[col] = df[col].astype(float)

    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')

    log_step("Computing prep time...")
    # Time_Orderd/Time_Order_picked show up in the data as either HH:MM:SS or HH:MM
    t_order = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce').fillna(
        pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce'))
    t_pick = pd.to_datetime(df['Time_Order_picked'], format='%H:%M:%S', errors='coerce').fillna(
        pd.to_datetime(df['Time_Order_picked'], format='%H:%M', errors='coerce'))

    df['Prep_Time_min'] = (t_pick - t_order).dt.total_seconds() / 60.0
    # Handle midnight crossovers (ordered at 23:55, picked up at 00:05)
    df['Prep_Time_min'] = np.where(df['Prep_Time_min'] < 0, df['Prep_Time_min'] + 1440, df['Prep_Time_min'])
    # Cap extreme outliers and fill missing with median
    df['Prep_Time_min'] = df['Prep_Time_min'].clip(upper=120)
    df['Prep_Time_min'] = df['Prep_Time_min'].fillna(df['Prep_Time_min'].median())

    log_step("Building cyclical time features and 168-hour buckets...")
    df['Hour'] = t_order.dt.hour.fillna(19.0)  # default to 19:00 if hour is missing
    df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
    df['Time_Bucket_168'] = df['Day_of_Week'].astype(str) + "_" + df['Hour'].astype(str)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

    log_step("Processing driver and vehicle fields...")
    df['Ratings_Category'] = pd.qcut(df['Delivery_person_Ratings'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['Ratings_Category'] = df['Ratings_Category'].cat.add_categories('Missing').fillna('Missing')
    df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median())
    df['multiple_deliveries'] = df['multiple_deliveries'].fillna(0)
    df['Vehicle_condition'] = df['Vehicle_condition'].fillna(df['Vehicle_condition'].median())

    # Driver stats are computed on train rows only, to avoid leaking test data into the profile
    train_only = df[df['is_train'] == 1].copy()
    driver_stats = train_only.groupby('Delivery_person_ID')['Time_taken(min)'].mean()
    fast_drivers = driver_stats[driver_stats < driver_stats.quantile(0.05)].index
    slow_drivers = driver_stats[driver_stats > driver_stats.quantile(0.95)].index

    df['Driver_Speed_Profile'] = df['Delivery_person_ID'].apply(
        lambda x: 'Fast_Outlier' if x in fast_drivers else ('Slow_Outlier' if x in slow_drivers else 'Normal')
    )

    log_step("Data processing complete.", t0)
    return df


# Spatial graph construction (H3 grid + edge aggregation)

def build_spatial_graph(df, h3_resolution=8):
    t0 = time.time()
    log_step(f"Mapping coordinates to H3 cells (resolution {h3_resolution})...")

    # h3 v4 renamed geo_to_h3 to latlng_to_cell; support both so this runs on either version
    try:
        df['Rest_H3'] = df.apply(lambda r: h3.latlng_to_cell(r['Restaurant_latitude'], r['Restaurant_longitude'], h3_resolution), axis=1)
        df['Del_H3'] = df.apply(lambda r: h3.latlng_to_cell(r['Delivery_location_latitude'], r['Delivery_location_longitude'], h3_resolution), axis=1)
    except AttributeError:
        df['Rest_H3'] = df.apply(lambda r: h3.geo_to_h3(r['Restaurant_latitude'], r['Restaurant_longitude'], h3_resolution), axis=1)
        df['Del_H3'] = df.apply(lambda r: h3.geo_to_h3(r['Delivery_location_latitude'], r['Delivery_location_longitude'], h3_resolution), axis=1)

    # A directed edge between the restaurant's cell and the delivery's cell
    df['Spatial_Edge'] = df['Rest_H3'] + "->" + df['Del_H3']

    log_step("Computing Haversine and Manhattan distances...")
    df['Edge_Distance_km'] = haversine_distance(
        df['Restaurant_latitude'], df['Restaurant_longitude'],
        df['Delivery_location_latitude'], df['Delivery_location_longitude']
    )
    df['Edge_Distance_Manhattan_km'] = manhattan_distance(
        df['Restaurant_latitude'], df['Restaurant_longitude'],
        df['Delivery_location_latitude'], df['Delivery_location_longitude']
    )

    log_step("Computing distance x traffic interaction feature...")
    traffic_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}
    df['Traffic_Multiplier'] = df['Road_traffic_density'].map(traffic_map).fillna(2)
    df['Effective_Distance'] = df['Edge_Distance_Manhattan_km'] * df['Traffic_Multiplier']

    log_step("Computing historical average delivery time per edge...")
    train_mask = df['is_train'] == 1
    edge_stats = df[train_mask].groupby('Spatial_Edge')['Time_taken(min)'].mean().reset_index()
    edge_stats.columns = ['Spatial_Edge', 'Historical_Edge_Time']

    df = df.merge(edge_stats, on='Spatial_Edge', how='left')

    # Edges with no training history (cold starts) get imputed via KNN over raw coordinates
    unseen_mask = df['Historical_Edge_Time'].isna()
    if unseen_mask.sum() > 0:
        log_step(f"Imputing {unseen_mask.sum():,} cold-start edges via KNN...")
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        known_df = df[~df['Historical_Edge_Time'].isna()]

        knn.fit(known_df[['Delivery_location_latitude', 'Delivery_location_longitude']],
                known_df['Historical_Edge_Time'])

        df.loc[unseen_mask, 'Historical_Edge_Time'] = knn.predict(
            df.loc[unseen_mask, ['Delivery_location_latitude', 'Delivery_location_longitude']]
        )

    log_step("Spatial graph complete.", t0)

    train = df[df['is_train'] == 1].copy()
    test = df[df['is_train'] == 0].copy()
    return train, test


# Model training and inference

def run_pipeline(train_path, test_path, output_path='submission.csv'):
    pipeline_start = time.time()

    df = process_data(train_path, test_path)
    train, test = build_spatial_graph(df, h3_resolution=8)

    num_features = [
        'Delivery_person_Age', 'Edge_Distance_km', 'Edge_Distance_Manhattan_km',
        'Effective_Distance', 'Historical_Edge_Time', 'Prep_Time_min',
        'Hour_Sin', 'Hour_Cos', 'Vehicle_condition', 'multiple_deliveries'
    ]

    cat_features = [
        'Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle',
        'Festival', 'City', 'Ratings_Category', 'Driver_Speed_Profile', 'Time_Bucket_168'
    ]

    log_step("Preparing categorical columns for CatBoost...")
    for col in cat_features:
        train[col] = train[col].astype(str).replace('nan', 'Missing')
        test[col] = test[col].astype(str).replace('nan', 'Missing')

    X_train_full = train[num_features + cat_features]
    y_train_full = train['Time_taken(min)']
    X_test = test[num_features + cat_features]

    # Hold out a validation split for evaluation
    log_step("Creating 80/20 train-validation split...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    log_step("Training CatBoost evaluation model...")
    t_val = time.time()

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='R2',
        cat_features=cat_features,
        verbose=False,
        random_seed=42
    )

    model.fit(X_train_split, y_train_split,
              eval_set=(X_val, y_val),
              early_stopping_rounds=50)

    val_predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, val_predictions)
    mse = mean_squared_error(y_val, val_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, val_predictions)

    log_step("Validation scoring complete.", t_val)
    print("\n" + "=" * 50)
    print("VALIDATION METRICS:")
    print(f"   Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"   Mean Squared Error (MSE):      {mse:.4f}")
    print(f"   Root Mean Squared Error (RMSE):{rmse:.4f}")
    print(f"   R-squared (R2) Score:          {r2:.4f}")
    print("=" * 50 + "\n")

    # Retrain on the full training set for the final submission model
    log_step("Retraining on 100% of training data...")
    t_final = time.time()
    model.fit(X_train_full, y_train_full)
    log_step("Final model training complete.", t_final)

    # Generate predictions and write submission file
    log_step(f"Writing predictions to '{output_path}'...")
    test['Time_taken (min)'] = model.predict(X_test)

    submission = test[['ID', 'Time_taken (min)']]
    submission.to_csv(output_path, index=False)

    log_step(f"Pipeline complete. Wrote {len(submission):,} rows to '{output_path}'.", pipeline_start)


if __name__ == "__main__":
    # Ensure train.csv and test.csv are in the same directory as this script
    run_pipeline('train.csv', 'test.csv')
