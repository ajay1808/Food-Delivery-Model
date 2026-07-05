# City-Scale ETA Prediction & Routing Engine (v1.0)

ML pipeline for predicting food delivery ETA. Instead of treating the problem as a flat regression over raw lat/lon and timestamps, it models the delivery network as a spatiotemporal graph: restaurant/delivery points are quantized onto Uber's H3 hex grid, routes carry historical travel-time estimates, and time is encoded cyclically instead of linearly.

## The Dataset

[Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset/) by Gaurav Malik, via Kaggle — ~45,000 delivery records across multiple cities. `archive.zip` in this repo is the unmodified download from that dataset page (`train.csv`, `test.csv`, `Sample_Submission.csv`); see the Kaggle page for its license terms.

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

#### H3 hex grid

Raw `(lat, lon)` pairs are a problem for this kind of aggregation: each delivery has its own unique coordinates, so there's no way to group deliveries by "location" and compute something like average delivery time for a neighborhood — every row is its own island.

[H3](https://h3geo.org/) is a geospatial indexing system (originally built at Uber for ride/delivery dispatch) that solves this by tiling the earth's surface into a grid of hexagonal cells and giving each cell a unique ID string. Snapping a raw coordinate to its enclosing cell turns a continuous, effectively infinite coordinate space into a finite, reusable set of locations — many different deliveries that happen to land in the same neighborhood now map to the same cell ID, so they can be grouped and aggregated.

Hexagons are used instead of a simple lat/lon grid of squares because every hexagon has 6 neighboring cells at equal distance from its center. A square grid doesn't have that property — a diagonal neighbor is farther away than an edge neighbor — which distorts any distance or adjacency calculation done on the grid itself.

H3 also supports multiple resolutions (0 = coarsest, continent-sized cells, up to 15 = finest, sub-meter cells). This pipeline uses resolution 8, where each cell averages roughly 0.7 km² — small enough to distinguish neighborhoods within a city, large enough that the same cell gets reused across many orders.

Concretely, in `build_spatial_graph`:

- `Rest_H3` and `Del_H3` are the hex cell IDs for the restaurant and the drop-off point.
- `Spatial_Edge` (`Rest_H3 -> Del_H3`) represents a directed route between two neighborhoods.
- `Historical_Edge_Time` is the mean delivery time across all training orders that share that same edge — a learned prior for "how long a delivery from neighborhood A to neighborhood B usually takes," computed on train only to avoid leakage.
- If a specific edge never appears in training (a route between two neighborhoods with no historical orders), there's nothing to look up. A KNN regressor (k=5, distance-weighted) over the raw delivery coordinates fills in an estimate from the nearest edges that do have historical data.

### Time features

- `Hour_Sin` / `Hour_Cos`: cyclical encoding of order hour, so hour 23 and hour 0 are adjacent instead of maximally far apart.
- `Time_Bucket_168`: day-of-week × hour bucket (e.g. `Friday_19`) to capture weekly demand patterns like the Friday dinner rush.
- `Day_of_Week` is also passed on its own, in addition to `Time_Bucket_168`, so the model can learn coarser weekday effects even where a specific day-hour bucket is sparse.

### Driver profiling

- `Driver_Speed_Profile`: using train-only per-driver average delivery time, drivers below the 5th percentile or above the 95th are tagged `Fast_Outlier` / `Slow_Outlier`, otherwise `Normal`.
- `Delivery_person_Ratings` is passed to the model directly as a numeric feature, in addition to the `Ratings_Category` quartile bucket. It's the single strongest raw correlate of delivery time in this dataset (r ≈ -0.34) — stronger than trip distance — and the quartile bucket alone discards most of that signal.

## Model

CatBoost over XGBoost, for two reasons:

1. Native categorical handling — the feature set is mostly categorical (weather, traffic, city, time bucket). CatBoost's ordered target encoding avoids the label-encoding step XGBoost would otherwise need.
2. Symmetric tree structure acts as regularization against noisy, one-off events in the data (e.g. flat tires, mispicks) that don't generalize.

## Results

Validation split (80/20), compared against a tuned XGBoost baseline on the same dataset. Numbers below are reproduced directly by running `eta_pipeline.py` in this repo:

| Metric | XGBoost | CatBoost + Network |
|---|---|---|
| MAE | 3.14 min | 3.04 min |
| RMSE | 3.93 min | 3.80 min |
| R² | 0.820 | 0.8355 |

## Usage

```bash
pip install pandas numpy h3 scikit-learn catboost
unzip archive.zip
python eta_pipeline.py
```

`archive.zip` is included in this repo; unzipping it produces `train.csv` and `test.csv` in the working directory, which is where the script expects them. `ETA_Prediction_Storytelling.ipynb` runs the same steps and includes an `!unzip archive.zip` cell at the top. The pipeline:

1. Cleans and feature-engineers the raw data (`process_data`)
2. Builds the H3 spatial graph and imputes cold-start edges (`build_spatial_graph`)
3. Trains/validates a CatBoost model on an 80/20 split, prints metrics
4. Retrains on 100% of the training data and writes `submission.csv`

## Future Work (v2.0)

V1 predicts ETA for a fixed route and a fixed price. V2 extends this into a simulation: detect when conditions deviate from what the model expects, reroute and reprice around the disruption, and negotiate the final delivery fee between two agents instead of hard-coding it.

### 1. Anomaly detection

- Simulate a stream of incoming orders, then inject a localized disruption (e.g. a blocked route inside a specific H3 cell).
- For each order, compare the V1 model's predicted ETA against a simulated actual delivery time. Track the residual (`actual − predicted`).
- Maintain a rolling interquartile range (IQR) of residuals. If orders in the same H3 cell start showing residuals greater than 1.5× IQR, flag that cell as anomalous.

### 2. Rerouting and pricing

- Once a cell is flagged, recompute the route to avoid it — this increases `Edge_Distance_km` for orders passing through that area.
- Compute a base fare from a fixed formula, e.g. `$2.00 + $0.50 × distance + $0.10 × predicted time`.
- Apply a surge multiplier (1.5x–2.5x) on top of the base fare to set a maximum budget for that delivery, rather than charging the multiplier directly.

### 3. Negotiation between two agents

- Instead of the price being fixed at the surge-adjusted number, two LLM agents (LangGraph or CrewAI) negotiate the final price within that budget.
- Dispatcher agent (the platform): can check the surge budget through a tool, and opens with a low offer to protect margin.
- Driver agent (the worker): can evaluate route profitability through a tool (detour distance, traffic, offered price), and counters with a reasoned rejection if the offer doesn't cover the added distance/time.
- The two exchange offers and counter-offers until they agree or hit a negotiation ceiling. The final agreed price is logged along with the reasoning that produced it.

### 4. Measuring negotiating power

For every simulated negotiation, quantify which side held more leverage rather than describing it qualitatively:

- Let `dispatcher_offer` be the dispatcher's opening offer, `driver_ask` be the driver's opening ask, and `final_price` be the agreed price.
- `driver_concession = (driver_ask − final_price) / (driver_ask − dispatcher_offer)`
- `dispatcher_concession = (final_price − dispatcher_offer) / (driver_ask − dispatcher_offer)`
- Whichever side conceded less (the smaller ratio) held more leverage in that negotiation.
- Aggregate this ratio across many simulated disruptions, broken out by conditions (surge multiplier, reroute distance, anomaly severity), to see when leverage shifts from the platform to the driver and under what circumstances.
