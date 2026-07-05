# City-Scale ETA Prediction & Routing Engine (v1.0)

ML pipeline for predicting food delivery ETA. Instead of treating the problem as a flat regression over raw lat/lon and timestamps, it models the delivery network as a spatiotemporal graph: restaurant/delivery points are quantized onto Uber's H3 hex grid, routes carry historical travel-time estimates, and time is encoded cyclically instead of linearly.

## The Dataset

[Food Delivery Dataset](https://www.kaggle.com/datasets) from Kaggle — ~45,000 delivery records across multiple cities.

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

## Feature Engineering

Raw columns are transformed into the following features.

### Prep time

`Time_taken` bundles driving time and restaurant wait time together. Separating them keeps the model from penalizing drivers for kitchen delays they don't control.

- `Prep_Time_min`: `Time_Order_picked` − `Time_Orderd`, with midnight-crossover correction and an upper clip to handle bad timestamps.

### Spatial features

Straight-line distance doesn't reflect how cars move through a city grid.

- `Edge_Distance_Manhattan_km`: L1-norm distance (with proper longitude scaling by `cos(lat)`), used as a closer approximation of city-grid driving than Haversine.
- `Effective_Distance`: Manhattan distance multiplied by a `Road_traffic_density` multiplier, as an interaction term for route friction.
- H3 hex grid (resolution 8): restaurant and delivery coordinates are mapped to discrete cells, forming an edge (`Rest_H3 -> Del_H3`) per delivery.
- `Historical_Edge_Time`: mean historical delivery time per edge, computed on train only. Edges not seen in training are imputed via a KNN regressor (k=5, distance-weighted) over delivery coordinates.

### Time features

- `Hour_Sin` / `Hour_Cos`: cyclical encoding of order hour, so hour 23 and hour 0 are adjacent instead of maximally far apart.
- `Time_Bucket_168`: day-of-week × hour bucket (e.g. `Friday_19`) to capture weekly demand patterns like the Friday dinner rush.

### Driver profiling

- `Driver_Speed_Profile`: using train-only per-driver average delivery time, drivers below the 5th percentile or above the 95th are tagged `Fast_Outlier` / `Slow_Outlier`, otherwise `Normal`.

## Model

CatBoost over XGBoost, for two reasons:

1. Native categorical handling — the feature set is mostly categorical (weather, traffic, city, time bucket). CatBoost's ordered target encoding avoids the label-encoding step XGBoost would otherwise need.
2. Symmetric tree structure acts as regularization against noisy, one-off events in the data (e.g. flat tires, mispicks) that don't generalize.

## Results

Validation split (80/20), compared against a tuned XGBoost baseline on the same dataset:

| Metric | Baseline (XGBoost) | This pipeline (CatBoost) |
|---|---|---|
| MAE | 3.14 min | ~2.91 min |
| RMSE | 3.93 min | ~3.75 min |
| R² | 0.820 | 0.835 – 0.842 |

## Usage

```bash
pip install pandas numpy h3 scikit-learn catboost
python eta_pipeline.py
```

Place `train.csv` and `test.csv` in the same directory as the script (or unzip the Kaggle `archive.zip`). The pipeline:

1. Cleans and feature-engineers the raw data (`process_data`)
2. Builds the H3 spatial graph and imputes cold-start edges (`build_spatial_graph`)
3. Trains/validates a CatBoost model on an 80/20 split, prints metrics
4. Retrains on 100% of the training data and writes `submission.csv`

## Future Work (v2.0)

- Multi-agent negotiation: a LangGraph/CrewAI setup where dispatcher agents (car vs. micro-mobility) negotiate routes in real time based on the ETAs from this model, to simulate fleet response to localized traffic shocks.
