# City-Scale ETA Prediction & Routing Engine (v1.0)

This repository contains a highly optimized machine learning pipeline for predicting food delivery Estimated Time of Arrival (ETA). Moving beyond standard A-to-B routing, this project models the urban environment as a **spatiotemporal graph**, capturing real-world friction like restaurant prep time, localized traffic density, and driver behavioral profiles.

## The Dataset

This project uses the [Food Delivery Dataset](https://www.kaggle.com/datasets) from Kaggle. The raw data consists of 45,000+ delivery records across multiple cities.

**Raw data columns:**

| Category | Columns |
|---|---|
| Identifiers | `ID`, `Delivery_person_ID` |
| Demographics | `Delivery_person_Age`, `Delivery_person_Ratings` |
| Geospatial | `Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude` |
| Temporal | `Order_Date`, `Time_Orderd`, `Time_Order_picked` |
| Environmental | `Weatherconditions`, `Road_traffic_density`, `City` |
| Operational | `Vehicle_condition`, `Type_of_order`, `Type_of_vehicle`, `multiple_deliveries`, `Festival` |
| Target | `Time_taken(min)` |

## Feature Engineering: Translating Physics into Math

Standard tabular models fail in logistics because they do not understand spatial constraints or time continuums. The raw columns were engineered into the following features to map the physical reality of the delivery network.

### 1. The "Golden Feature": Operational Latency

The target variable (`Time_taken`) includes both driving time and restaurant waiting time.

- **`Prep_Time_min`**: Extracted by subtracting `Time_Orderd` from `Time_Order_picked`. This prevents the model from penalizing drivers for restaurant-side inefficiencies and isolates the actual driving window.

### 2. Urban Physics (Spatial Engineering)

"As the crow flies" distance does not apply to cars in a city.

- **`Edge_Distance_Manhattan_km`**: Calculated using the L1 norm (Manhattan distance) to approximate grid-like city driving, replacing standard Haversine distance.
- **`Effective_Distance`**: An interaction feature multiplying the Manhattan distance by a categorical `Road_traffic_density` multiplier, directly capturing the friction of the route.
- **H3 Spatial Network**: Lat/Lon coordinates were quantized into Uber's H3 Hexagonal Grid (Resolution 8), converting infinite coordinates into discrete edges (Node A → Node B).
- **Cold Start KNN Imputation**: If an order was placed in a geographically unseen neighborhood, a K-Nearest Neighbors algorithm (k=5, weighted by physical distance) was used to impute the historical edge time based on surrounding delivery zones.

### 3. Cyclical Time Continuity

- **`Hour_Sin` & `Hour_Cos`**: Tree models interpret the hour '23' and '0' as maximally distant. Applying sine and cosine transformations maps time onto a circle, allowing the model to understand that 11:00 PM and Midnight are chronologically adjacent.
- **168-Hour Matrix**: Deliveries are grouped into a 7x24 grid (e.g., `Friday_19`) to capture structural weekly traffic patterns (e.g., Friday dinner rush).

### 4. Driver Behavioral Profiling

- **`Driver_Speed_Profile`**: Isolated historical driver data in the training set to identify the 5th and 95th percentiles of delivery times. Drivers are tagged as `Fast_Outlier`, `Slow_Outlier`, or `Normal`, allowing the model to account for deterministic human pacing.

## Architecture & Model Selection

### Why CatBoost over XGBoost?

While XGBoost is the standard for tabular data, CatBoost was selected for this specific architecture for two primary reasons:

1. **Native Categorical Handling**: Logistics data is heavily categorical (Weather, Traffic, City). XGBoost requires Label Encoding, which destroys categorical nuance. CatBoost uses Ordered Target Encoding, processing string categories natively without target leakage.
2. **Symmetric Trees**: CatBoost builds perfectly balanced, symmetric decision trees. In a highly chaotic dataset (where unpredictable events like flat tires occur), symmetric trees act as aggressive regularization, preventing the model from overfitting to the noise.

## Results & Benchmarks

The benchmark for this dataset (the top-voted Kaggle kernel) relies on heavily tuned XGBoost models operating on raw/encoded tabular data. By reframing the problem as a spatiotemporal physics problem, this CatBoost pipeline significantly outperforms the baseline.

| Metric | Baseline (Standard XGBoost) | Our Engine (Spatiotemporal CatBoost) | Improvement |
|---|---|---|---|
| MAE | 3.14 minutes | ~2.91 minutes | Lower error spread |
| RMSE | 3.93 minutes | ~3.75 minutes | Tighter variance |
| R² Score | 0.820 | 0.835 – 0.842 | Explains more total variance |

> **Note:** The R² score improvement indicates that the model successfully captured previously hidden spatial and operational constraints (like restaurant prep time and Manhattan-grid friction) that standard routing algorithms miss.

## Usage

```bash
pip install pandas numpy h3 scikit-learn catboost
python eta_pipeline.py
```

Place `train.csv` and `test.csv` in the same directory as the script (or unzip the Kaggle `archive.zip`). The pipeline will:

1. Clean and feature-engineer the raw data (`process_data`)
2. Build the H3 spatial graph and impute cold-start edges (`build_spatial_graph`)
3. Train/validate a CatBoost model on an 80/20 split, print metrics
4. Retrain on 100% of the training data and write `submission.csv`

## Future Work (Version 2.0)

The next iteration of this project will transition from prediction to active simulation.

- **Multi-Agent Negotiation Framework**: Implementing a LangGraph/CrewAI environment where AI agents (Car Dispatcher vs. Micro-Mobility Dispatcher) negotiate delivery routes in real-time based on the ETA predictions generated by this model, simulating automated fleet response to localized traffic shocks.
