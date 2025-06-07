# Prometheus Time Series Anomaly Detection with LSTM Autoencoder

This project implements a system for detecting anomalies in time series data collected from Prometheus. It uses an LSTM (Long Short-Term Memory) autoencoder model built with TensorFlow/Keras to learn normal patterns from your metrics and identify deviations. The system includes scripts for data collection, preprocessing, model training, and real-time anomaly detection, exposing results via a Prometheus exporter.

**GitHub Repository:** [https://github.com/vpuhoff/prometheus-anomaly-detection-lstm](https://github.com/vpuhoff/prometheus-anomaly-detection-lstm)

**PyPI Package:** [https://pypi.org/project/prometheus-anomaly-detection-lstm](https://pypi.org/project/prometheus-anomaly-detection-lstm)

## Features

* **Data Collection:** Fetches time series data from a Prometheus instance for specified PromQL queries.
* **Efficient Caching:** Caches historical data from Prometheus in small, reusable chunks to dramatically speed up subsequent runs and reduce redundant API calls.
* **Preprocessing:** Handles missing values and normalizes/scales values for optimal model training.
* **LSTM Autoencoder Training:** Trains an LSTM autoencoder on the full preprocessed dataset.
* **Real-time Anomaly Detection:** Continuously monitors new data and processes it with the trained model to detect anomalies.
* **Prometheus Exporter Integration:** Exposes key anomaly detection metrics (e.g., reconstruction error, anomaly flag) that can be scraped by Prometheus.
* **Simplified Workflow:** Uses `uv` for ultra-fast dependency management and a `Makefile` for easy, standardized command execution.
* **Configurable:** All stages are highly configurable via a central `config.yaml` file.

## Project Structure

```
.
├── artifacts/                  # Directory for all generated files (data, models, etc.)
│   └── prometheus_cache/       # On-disk cache for Prometheus queries
├── .github/
│   └── workflows/
│       └── publish.yml         # CI/CD workflow for publishing to PyPI
├── config.yaml                 # Central configuration file for all scripts
├── cli.py                      # Command-line utility to run workflow stages
├── data_collector.py           # Script to collect historical data
├── preprocess_data.py          # Script to preprocess the collected data
├── train_autoencoder.py        # Script to train the LSTM autoencoder
├── realtime_detector.py        # Script for real-time anomaly detection
├── pyproject.toml              # Project definition and dependencies (PEP 621)
├── requirements.lock.txt       # Locked versions of all dependencies
├── Makefile                    # Makefile for simplified command execution
└── README.md                   # This file
```

## Prerequisites

* Python 3.12 or later.
* `uv`, the fast Python package installer.
* `make` (available on most Linux/macOS systems).
* A running Prometheus instance (v2.x or later) that is scraping the metrics you want to analyze.

## Setup & Installation

The project uses `uv` for dependency management and a `Makefile` to simplify commands.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/vpuhoff/prometheus-anomaly-detection-lstm](https://github.com/vpuhoff/prometheus-anomaly-detection-lstm)
    cd prometheus-anomaly-detection-lstm
    ```

2.  **Run the automated setup:**
    This single command will automatically:
    * Install `uv` if it's not already present.
    * Create a virtual environment in `.venv/`.
    * Install all required dependencies from the lock file.

    ```bash
    make install
    ```

3.  **Activate the Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Configure `config.yaml`:**
The `config.yaml` file is central to running this project. Key sections include:

  * **`artifacts_dir`**: The directory where all generated artifacts (datasets, scalers, models, plots) will be saved. This helps to keep the main project directory clean.
  * **`prometheus_url`**: URL of your Prometheus server.
  * **`queries`**: Dictionary of PromQL queries with friendly aliases.
  * **`data_settings`**: Parameters for `data_collector.py`.
      * `collection_periods_iso`: (Recommended) A list of specific time ranges to collect data from. This is the best way to create a high-quality training dataset by explicitly including periods of known normal operation and excluding periods with anomalies. If this parameter is present, it will be used instead of the other time settings.
        ```yaml
        collection_periods_iso:
          - start: "2025-05-20T10:00:00"
            end: "2025-05-22T18:00:00"
          - start: "2025-05-25T09:00:00"
            end: "2025-05-27T12:00:00"
        ```
      * `collection_period_hours`, `start_time_iso`, `end_time_iso`: Legacy parameters for specifying a single data collection window. These are used only if `collection_periods_iso` is not defined.
      * `step`: Defines the data sampling interval (e.g., `30s`, `2m`).
      * `output_filename`: The name of the output Parquet file.
      * `cache_chunk_hours`: (Optional) The size in hours for splitting large time ranges into smaller chunks for more efficient caching. Defaults to `1`.
  * **`preprocessing_settings`**: Parameters for `preprocess_data.py` (e.g., `nan_fill_strategy`, `scaler_type`, `processed_output_filename`, `scaler_output_filename`).
  * **`training_settings`**: Parameters for `train_autoencoder.py`.
      * `model_output_filename`: Filename for the trained model.
      * `sequence_length`, `train_split_ratio`, `epochs`, `batch_size`, `learning_rate`, `early_stopping_patience`: Standard training hyperparameters.
      * `lstm_units_encoder1`, etc.: LSTM autoencoder architecture definition.
  * **`data_filtering_settings`**: Parameters for the optional `filter_anomalous_data.py` script.
      * `normal_sequences_output_filename`: Output file for sequences classified as normal.
      * `anomalous_sequences_output_filename`: Output file for sequences classified as anomalous.
  * **`real_time_anomaly_detection`**: Parameters for `realtime_detector.py`.
      * `query_interval_seconds`: How often to fetch new data.
      * `anomaly_threshold_mse`: **Crucial\!** MSE threshold for declaring an anomaly. Tune this based on the error histogram generated during training.
      * `exporter_port`: Port for the Prometheus exporter.
      * `metrics_prefix`: Prefix for exposed Prometheus metrics.

**Before running any script, review and customize `config.yaml` thoroughly.**

## Usage / Workflow

The project follows a sequential workflow. Each stage can be launched via the `cli.py` utility. All output files will be placed in the directory specified by `artifacts_dir` in `config.yaml`.

```bash
python cli.py collect       # сбор данных
python cli.py preprocess    # предобработка
python cli.py train         # обучение модели
python cli.py detect        # запуск realtime детектора
```

The sequential workflow is as follows:

**Step 1: Data Collection (`data_collector.py`)**
Collect historical data from your Prometheus instance. This script can combine data from multiple time ranges if specified in `config.yaml` under `collection_periods_iso`. The script uses an efficient caching mechanism, so the first run might be slow, but subsequent runs for the same time periods will be significantly faster.

```bash
python data_collector.py
```

Output: Raw data Parquet file (e.g., `prometheus_metrics_data.parquet`) which includes `day_of_week` and `hour_of_day` columns, saved in the `artifacts_dir` directory. A `prometheus_cache` subdirectory will also be created here.

**Step 2: Data Preprocessing (`preprocess_data.py`)**
Preprocess the collected data (handles NaNs, scales features).

```bash
python preprocess_data.py
```

Outputs: A processed data Parquet file (e.g., `processed_metrics_data.parquet`) and a saved scaler (e.g., `fitted_scaler.joblib`), both saved in `artifacts_dir`.

**Step 3: Train Model (`train_autoencoder.py`)**
Train the LSTM autoencoder on the entire preprocessed dataset from Step 2.

```bash
python train_autoencoder.py
```

Outputs (all saved in `artifacts_dir`):

  * A trained Keras model (e.g., `lstm_autoencoder_model.keras`).
  * A training history plot (`training_history_loss_...png`).
  * A reconstruction error histogram (`reconstruction_error_histogram_...png`). **Use this histogram to determine an appropriate value for `anomaly_threshold_mse` in `config.yaml`**.

**Step 4: Real-time Anomaly Detection (`realtime_detector.py`)**
Run the real-time detector using the trained model from Step 3.

  * Ensure `model_output_filename` in `training_settings` points to your trained model.
  * Ensure `anomaly_threshold_mse` in `real_time_anomaly_detection` is correctly set based on the histogram from Step 3.
  * The script will automatically look for the model and scaler in the `artifacts_dir` directory.

```bash
python realtime_detector.py
```

All primary commands are managed through the `Makefile` for simplicity and consistency.

**Main Workflow Steps:**
The project follows a sequential workflow. Run these commands in order:

```bash
# 1. Collect historical data from Prometheus
make collect

# 2. Preprocess the collected data (scaling, NaN handling)
make preprocess

# 3. Train the LSTM autoencoder model
make train

# 4. Start the real-time detector with a Prometheus exporter
make detect
```

**Dependency Management:**

* **To add or change a dependency:**
    1.  Edit the `dependencies` or `dev` section in `pyproject.toml`.
    2.  Run `make update`. This will update `requirements.lock.txt` and sync your environment.

* **To sync your environment after pulling changes:**
    If `requirements.lock.txt` was updated in the repository, just run:
    ```bash
    make sync
    ```

To see all available commands, run `make help`.

## Monitoring (Prometheus & Grafana)

Configure Prometheus to scrape the metrics endpoint exposed by the real-time detector (the address is specified in `config.yaml`, e.g., `http://localhost:8901/metrics`).

Key metrics to monitor:

* `anomaly_detector_latest_reconstruction_error_mse`
* `anomaly_detector_is_anomaly_detected`
* `anomaly_detector_total_anomalies_count_total`
* `anomaly_detector_feature_reconstruction_error_mse{feature_name="your_alias"}`

## Interpreting Results

* **Monitoring Metrics:** Observe the `is_anomaly_detected` and `latest_reconstruction_error_mse` metrics in real time to evaluate detection behavior.
* **Per-Feature Errors:** When an anomaly is flagged, check the corresponding `feature_reconstruction_error_mse` metrics to see which specific time series are contributing most to the anomaly.

## Customization & Extending

  * **Monitoring New Metrics:** Add new PromQL queries to `config.yaml`. Retrain the model (run steps 2-3) to include these new features.
  * **Tuning Anomaly Threshold:** The `anomaly_threshold_mse` value is critical. Adjust it based on the training error histogram and desired sensitivity.
  * **Model Architecture:** Modify LSTM parameters in the `training_settings` section of `config.yaml`.

## Troubleshooting

* **`make` command not found:** Install `make` using your system's package manager (e.g., `sudo apt-get install build-essential` on Debian/Ubuntu).
* **Prometheus Connection:** Verify `prometheus_url` and query validity in `config.yaml`.
  * **Data Issues:** Check for "No data found" errors; inspect PromQL queries and Prometheus scrape targets. Review `nan_fill_strategy` if NaNs persist.
  * **Model Training:** If loss doesn't decrease, adjust learning rate, batch size, or architecture. `EarlyStopping` is configured to prevent overfitting.
  * **File Not Found:** Double-check filenames in `config.yaml`. Ensure that the `artifacts_dir` setting is correct and that the necessary input files exist in that directory.
  * **Forcing Data Re-fetch:** If you need to force the system to re-download data from Prometheus and ignore the cache, you can manually delete the `prometheus_cache` directory inside your `artifacts_dir`.
  * **Port in Use:** If `realtime_detector.py` fails, the `exporter_port` might be occupied by another process.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.