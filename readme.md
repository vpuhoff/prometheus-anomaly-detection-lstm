# Prometheus Time Series Anomaly Detection with LSTM Autoencoder

This project implements a system for detecting anomalies in time series data collected from Prometheus. It uses an LSTM (Long Short-Term Memory) autoencoder model built with TensorFlow/Keras to learn normal patterns from your metrics and identify deviations. The system includes scripts for data collection, preprocessing, model training (including an optional second model trained on "cleaned" data), data filtering, and real-time anomaly detection with two models running concurrently, exposing their results via a Prometheus exporter.

## Features

* **Data Collection:** Fetches time series data from a Prometheus instance for specified PromQL queries.
* **Preprocessing:** Handles missing values and normalizes/scales data for optimal model training.
* **LSTM Autoencoder Training (Two-Model Strategy):**
    * Trains an initial LSTM autoencoder (Model A) on the full preprocessed dataset.
    * Optionally trains a second LSTM autoencoder (Model B) on a "cleaned" dataset, from which anomalies identified by Model A have been filtered out.
* **Data Filtering:** A script to apply the trained Model A to filter out anomalous sequences from a dataset.
* **Real-time Anomaly Detection:** Continuously monitors new data, preprocesses it, and uses *both* trained models (Model A and Model B) simultaneously to detect anomalies.
* **Prometheus Exporter Integration:** Exposes key anomaly detection metrics separately for each model (e.g., reconstruction error, anomaly flag, per-feature errors) that can be scraped by Prometheus and monitored with tools like Grafana.
* **Configurable:** All stages are highly configurable via a central `config.yaml` file.

## Project Structure

```
.
├── config.yaml                 # Central configuration file for all scripts
├── data_collector.py           # Script to collect historical data from Prometheus
├── preprocess_data.py          # Script to preprocess the collected data
├── train_autoencoder.py        # Script to train LSTM autoencoder models (handles both initial and cleaned data training)
├── filter_anomalous_data.py    # Script to filter data using a trained model to separate normal/anomalous sequences
├── realtime_detector.py        # Script for real-time anomaly detection using two models and Prometheus exporter
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Prerequisites

* Python 3.8+
* Pip (Python package installer)
* A running Prometheus instance (v2.x or later) that is scraping the metrics you want to analyze.
* (Optional) Exporters configured for your Prometheus to collect the desired metrics (e.g., `node_exporter`, `windows_exporter`).

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    A `requirements.txt` file should be included in the repository. Install dependencies using:
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not present, you can create one after installing the necessary packages manually:
    ```bash
    pip install pandas numpy PyYAML scikit-learn tensorflow joblib requests prometheus_client matplotlib
    pip freeze > requirements.txt
    ```

4.  **Prometheus Setup:**
    Ensure your Prometheus server is running and accessible. The scripts will query this server based on the URL and PromQL queries defined in `config.yaml`. The example queries in `config.yaml` might use metrics from `windows_exporter`; adapt these to your own available metrics.

## Configuration (`config.yaml`)

The `config.yaml` file is central to running this project. Key sections include:

* **`prometheus_url`**: URL of your Prometheus server.
* **`queries`**: Dictionary of PromQL queries with friendly aliases.
* **`data_settings`**: Parameters for `data_collector.py` (e.g., `collection_period_hours`, `step`, `output_filename`).
* **`preprocessing_settings`**: Parameters for `preprocess_data.py` (e.g., `nan_fill_strategy`, `scaler_type`, `processed_output_filename`, `scaler_output_filename`).
* **`training_settings`**: Parameters for `train_autoencoder.py`.
    * `model_output_filename`: Filename for Model A (trained on all data).
    * `sequence_length`, `train_split_ratio`, `epochs`, `batch_size`, `learning_rate`, `early_stopping_patience`: Standard training hyperparameters.
    * `lstm_units_encoder1`, etc.: LSTM autoencoder architecture definition.
    * `train_on_filtered_sequences`: Set to `true` to train Model B.
    * `filtered_normal_sequences_input_filename`: Input `.npy` file of normal sequences for training Model B (typically output from `filter_anomalous_data.py`).
    * `filtered_model_output_filename`: Filename for Model B (trained on cleaned data).
* **`data_filtering_settings`**: Parameters for `filter_anomalous_data.py`.
    * `normal_sequences_output_filename`: Output file for sequences classified as normal by Model A.
    * `anomalous_sequences_output_filename`: Output file for sequences classified as anomalous by Model A.
* **`real_time_anomaly_detection`**: Parameters for `realtime_detector.py`.
    * `query_interval_seconds`: How often to fetch new data.
    * `anomaly_threshold_mse_model_a`: **Crucial!** MSE threshold for Model A. Tune based on Model A's validation error histogram.
    * `anomaly_threshold_mse_model_b`: **Crucial!** MSE threshold for Model B. Tune based on Model B's validation error histogram.
    * `exporter_port`: Port for the Prometheus exporter.
    * `metrics_prefix`: Prefix for exposed Prometheus metrics.

**Before running any script, review and customize `config.yaml` thoroughly.**

## Usage / Workflow

The project follows a sequential workflow:

**Step 1: Data Collection (`data_collector.py`)**
Collect historical data from your Prometheus instance.
```bash
python data_collector.py
```
Output: Raw data Parquet file (e.g., `prometheus_metrics_data.parquet`).

**Step 2: Data Preprocessing (`preprocess_data.py`)**
Preprocess the collected data (handles NaNs, scales features).
```bash
python preprocess_data.py
```
Outputs: Processed data Parquet file (e.g., `processed_metrics_data.parquet`) and a saved scaler (e.g., `fitted_scaler.joblib`).

**Step 3: Train Initial Model - Model A (`train_autoencoder.py`)**
Train the first LSTM autoencoder (Model A) on the full preprocessed dataset.
* In `config.yaml` (`training_settings`):
    * Set `train_on_filtered_sequences: false`.
    * Configure `model_output_filename` (e.g., `lstm_autoencoder_model_A.keras`).
```bash
python train_autoencoder.py
```
Outputs: Trained Model A (e.g., `lstm_autoencoder_model_A.keras`), training history plots. Use the `reconstruction_error_histogram_...A.png` to help determine `anomaly_threshold_mse_model_a` in `config.yaml`.

**Step 4: Filter Data (Optional, for Model B) (`filter_anomalous_data.py`)**
Use the trained Model A to classify sequences in your preprocessed dataset as "normal" or "anomalous".
* Ensure `anomaly_threshold_mse` (from `real_time_anomaly_detection` section, used by this script as the threshold for Model A) is appropriately set in `config.yaml`.
* Configure output filenames in `data_filtering_settings`.
```bash
python filter_anomalous_data.py
```
Outputs: `.npy` files for normal sequences (e.g., `normal_sequences.npy`) and anomalous sequences.

**Step 5: Train Cleaned Model - Model B (Optional) (`train_autoencoder.py`)**
Train the second LSTM autoencoder (Model B) on the "normal" sequences identified in Step 4.
* In `config.yaml` (`training_settings`):
    * Set `train_on_filtered_sequences: true`.
    * Set `filtered_normal_sequences_input_filename` to the output from Step 4 (e.g., `normal_sequences.npy`).
    * Configure `filtered_model_output_filename` (e.g., `lstm_autoencoder_model_B_cleaned.keras`).
    * Ensure other training parameters (architecture, epochs etc.) are set for a fair comparison if desired.
```bash
python train_autoencoder.py
```
Outputs: Trained Model B (e.g., `lstm_autoencoder_model_B_cleaned.keras`), training history plots. Use its `reconstruction_error_histogram_...B_cleaned.png` to help determine `anomaly_threshold_mse_model_b` in `config.yaml`.

**Step 6: Real-time Anomaly Detection (`realtime_detector.py`)**
Run the real-time detector, which now uses both Model A and Model B concurrently.
* Ensure `model_output_filename` (for Model A) and `filtered_model_output_filename` (for Model B) in `training_settings` point to your trained models.
* Ensure `anomaly_threshold_mse_model_a` and `anomaly_threshold_mse_model_b` in `real_time_anomaly_detection` are correctly set.
```bash
python realtime_detector.py
```
The detector starts a Prometheus exporter (e.g., on `http://localhost:8001/metrics`).

**Step 7: Monitoring (Prometheus & Grafana)**
Configure Prometheus to scrape the metrics endpoint from `realtime_detector.py`. Visualize metrics like:
* `anomaly_detector_latest_reconstruction_error_mse{model_id="A"}`
* `anomaly_detector_latest_reconstruction_error_mse{model_id="B"}`
* `anomaly_detector_is_anomaly_detected{model_id="A"}`
* `anomaly_detector_is_anomaly_detected{model_id="B"}`
* `anomaly_detector_total_anomalies_count_total{model_id="A"}`
* `anomaly_detector_total_anomalies_count_total{model_id="B"}`
* `anomaly_detector_feature_reconstruction_error_mse{model_id="A", feature_name="your_alias"}`
* `anomaly_detector_feature_reconstruction_error_mse{model_id="B", feature_name="your_alias"}`

## Interpreting Results

* **Compare Models:** Observe the `is_anomaly_detected` and `latest_reconstruction_error_mse` metrics for both Model A and Model B in real-time. This helps understand if training on "cleaned" data (Model B) yields different or more desirable detection behavior.
* **Per-Feature Errors:** When an anomaly is flagged by either model, check the corresponding `feature_reconstruction_error_mse` metrics (and logs of `realtime_detector.py`) to see which specific time series (features) are contributing most to the anomaly.

## Customization & Extending

* **Monitoring New Metrics:** Add PromQL queries to `config.yaml`. Retrain models (all relevant steps) to include these.
* **Tuning Anomaly Thresholds:** The `anomaly_threshold_mse_model_a` and `anomaly_threshold_mse_model_b` are critical. Adjust them based on model performance and desired sensitivity.
* **Model Architecture:** Modify LSTM parameters in `training_settings` of `config.yaml`.
* **Experimentation:** Use the `filter_anomalous_data.py` script with different thresholds for Model A to generate various "cleaned" datasets for training Model B.

## Troubleshooting

* **Python Dependencies:** Ensure `requirements.txt` is up-to-date and packages are installed.
* **Prometheus Connection:** Verify `prometheus_url` and query validity.
* **Data Issues:** Check for "No data found" errors; inspect PromQL queries and Prometheus scrape targets. Review `nan_fill_strategy` if NaNs persist.
* **Model Training:** If loss doesn't decrease, adjust learning rate, batch size, or architecture. For overfitting, utilize `EarlyStopping` or consider more data/regularization.
* **File Not Found:** Double-check filenames in `config.yaml` against actual generated files (models, scalers, datasets).
* **Port in Use:** If `realtime_detector.py` fails, the `exporter_port` might be occupied.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.