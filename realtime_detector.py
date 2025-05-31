import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import time
import requests
from datetime import datetime, timedelta
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib # For scaler

from prometheus_client import start_http_server, Gauge, Counter, REGISTRY

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Глобальные переменные для Prometheus метрик (ИСПРАВЛЕННЫЕ ИМЕНА) ---
PROM_LATEST_RECONSTRUCTION_ERROR_MSE = None
PROM_IS_ANOMALY_DETECTED = None
PROM_TOTAL_ANOMALIES_COUNT = None
PROM_FEATURE_RECONSTRUCTION_ERROR_MSE = None
PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS = None
PROM_DATA_POINTS_IN_CURRENT_WINDOW = None


class RealtimeAnomalyDetector:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        
        self.prom_url = self.config.get('prometheus_url')
        self.queries = self.config.get('queries', {})
        self.metric_columns_ordered = list(self.queries.keys()) 

        rt_config = self.config.get('real_time_anomaly_detection', {})
        self.query_interval = rt_config.get('query_interval_seconds', 60)
        self.exporter_port = rt_config.get('exporter_port', 8001)
        self.metrics_prefix = rt_config.get('metrics_prefix', 'anomaly_detector_')

        self.anomaly_threshold_a = rt_config.get('anomaly_threshold_mse_model_a', 0.0025)
        self.anomaly_threshold_b = rt_config.get('anomaly_threshold_mse_model_b', 0.0020)

        preprocess_config = self.config.get('preprocessing_settings', {})
        training_config = self.config.get('training_settings', {})
        
        scaler_filename = preprocess_config.get('scaler_output_filename', 'fitted_scaler.joblib')
        model_a_filename = training_config.get('model_output_filename', 'lstm_autoencoder_model_A.keras')
        model_b_filename = training_config.get('filtered_model_output_filename', 'lstm_autoencoder_model_B_cleaned.keras')
        self.sequence_length = training_config.get('sequence_length', 20)

        data_s_config = self.config.get('data_settings',{})
        self.data_step_duration_str = rt_config.get('data_step_duration', data_s_config.get('step', '30s'))

        base_dir = Path(__file__).resolve().parent
        self.scaler_path = base_dir / scaler_filename
        self.model_a_path = base_dir / model_a_filename
        self.model_b_path = base_dir / model_b_filename

        self.scaler = self._load_scaler()
        logging.info("Загрузка Модели A...")
        self.model_a = self._load_tf_model(self.model_a_path, "Модель A")
        logging.info("Загрузка Модели B...")
        self.model_b = self._load_tf_model(self.model_b_path, "Модель B")
        
        if self.scaler:
            self.num_features = self.scaler.n_features_in_
            if self.num_features != len(self.queries):
                logging.error(f"Расхождение в количестве признаков! 'queries': {len(self.queries)}, Скейлер: {self.num_features}.")
        else:
            logging.error("Скейлер не загружен.")
            self.num_features = len(self.queries)

        self._setup_prometheus_metrics()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            logging.info(f"Конфигурация успешно загружена из {self.config_path}")
            return config_data
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации {self.config_path}: {e}", exc_info=True)
            exit(1)

    def _load_scaler(self):
        if not self.scaler_path.exists():
            logging.error(f"Файл скейлера не найден: {self.scaler_path}")
            return None
        try:
            scaler = joblib.load(self.scaler_path)
            logging.info(f"Скейлер успешно загружен из {self.scaler_path}")
            return scaler
        except Exception as e:
            logging.error(f"Ошибка загрузки скейлера из {self.scaler_path}: {e}", exc_info=True)
            return None

    def _load_tf_model(self, model_path: Path, model_name_log: str):
        if not model_path.exists():
            logging.warning(f"Файл модели '{model_name_log}' не найден: {model_path}.")
            return None
        try:
            model = load_model(model_path)
            logging.info(f"Информация о {model_name_log}:")
            model.summary(print_fn=logging.info)
            logging.info(f"{model_name_log} успешно загружена из {model_path}")
            return model
        except Exception as e:
            logging.error(f"Ошибка загрузки {model_name_log} из {model_path}: {e}", exc_info=True)
            return None

    def _td_seconds(self, td_str: str) -> int:
        if td_str.endswith('s'): return int(td_str[:-1])
        if td_str.endswith('m'): return int(td_str[:-1]) * 60
        if td_str.endswith('h'): return int(td_str[:-1]) * 3600
        try: return int(td_str)
        except ValueError:
            logging.warning(f"Не распознана длительность шага: {td_str}. Используется 30с.")
            return 30
            
    def _fetch_data_window(self) -> pd.DataFrame | None:
        if not self.prom_url or not self.queries:
            logging.error("URL Prometheus или запросы не определены.")
            return None
        step_seconds = self._td_seconds(self.data_step_duration_str)
        window_duration_seconds = self.sequence_length * step_seconds
        end_time = datetime.now()
        start_time_query = end_time - timedelta(seconds=window_duration_seconds + step_seconds * 2)
        all_metric_dfs = []
        logging.info(f"Запрос данных: {start_time_query.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}, шаг {self.data_step_duration_str}")
        for custom_name in self.metric_columns_ordered:
            query_string = self.queries[custom_name]
            api_url = f"{self.prom_url}/api/v1/query_range"
            params = {'query': query_string, 'start': start_time_query.timestamp(), 'end': end_time.timestamp(), 'step': self.data_step_duration_str}
            try:
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    if len(data['data']['result']) > 0:
                        values = data['data']['result'][0].get('values', [])
                        if values:
                            df_metric = pd.DataFrame(values, columns=['timestamp', custom_name]).set_index(pd.to_datetime(pd.DataFrame(values, columns=['timestamp', custom_name])['timestamp'], unit='s'))[[custom_name]]
                            df_metric[custom_name] = pd.to_numeric(df_metric[custom_name], errors='coerce')
                            all_metric_dfs.append(df_metric)
                        else: logging.warning(f"Нет значений для '{custom_name}'."); return None 
                    else: logging.warning(f"Пустой 'result' для '{custom_name}'."); return None
                else: logging.warning(f"Неуспех для '{custom_name}': {data.get('errorType', '')} {data.get('error', data.get('status'))}"); return None
            except Exception as e: logging.error(f"Ошибка для '{custom_name}': {e}", exc_info=True); return None
        if not all_metric_dfs or len(all_metric_dfs) != len(self.queries): logging.warning("Не все метрики загружены."); return None
        try: final_df = pd.concat(all_metric_dfs, axis=1, join='inner') 
        except Exception as e: logging.error(f"Ошибка объединения DataFrame: {e}", exc_info=True); return None
        if final_df.empty: logging.warning("Итоговый DataFrame пуст."); return None
        try: final_df = final_df[self.metric_columns_ordered]
        except KeyError as e: logging.error(f"Ошибка порядка колонок: {e}. Ожид: {self.metric_columns_ordered}. Получ: {final_df.columns.tolist()}"); return None
        if len(final_df) >= self.sequence_length: return final_df.tail(self.sequence_length)
        else:
            logging.warning(f"Недостаточно данных ({len(final_df)}) для посл. ({self.sequence_length}).")
            if PROM_DATA_POINTS_IN_CURRENT_WINDOW: PROM_DATA_POINTS_IN_CURRENT_WINDOW.set(len(final_df))
            return None

    def _preprocess_and_create_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.scaler is None: logging.error("Скейлер не загружен."); return None
        if df.isnull().values.any():
            logging.warning("NaN в окне, применяем ffill().bfill()")
            df = df.sort_index().ffill().bfill()
            if df.isnull().values.any(): logging.error(f"NaN остались. Колонки: {df.columns[df.isnull().any()].tolist()}"); return None
        try:
            if df[self.metric_columns_ordered].shape[1] != self.num_features:
                 logging.error(f"Ошибка признаков. Ожид: {self.num_features}, есть: {df[self.metric_columns_ordered].shape[1]}."); return None
            scaled_values = self.scaler.transform(df[self.metric_columns_ordered].values)
        except Exception as e: logging.error(f"Ошибка масштабирования: {e}. Данные (форма {df[self.metric_columns_ordered].shape}):\n{df[self.metric_columns_ordered].head(2)}"); return None
        return np.expand_dims(scaled_values, axis=0)

    def _setup_prometheus_metrics(self):
        global PROM_LATEST_RECONSTRUCTION_ERROR_MSE, PROM_IS_ANOMALY_DETECTED, \
               PROM_TOTAL_ANOMALIES_COUNT, PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS, \
               PROM_DATA_POINTS_IN_CURRENT_WINDOW, PROM_FEATURE_RECONSTRUCTION_ERROR_MSE
        
        model_id_label_list = ['model_id']
        feature_error_label_list = ['model_id', 'feature_name']
        metric_definitions = {
            'latest_reconstruction_error_mse': ('MSE ошибка реконструкции для последнего окна', model_id_label_list, Gauge),
            'is_anomaly_detected': ('Флаг аномалии (1 если аномалия, 0 если норма)', model_id_label_list, Gauge),
            'total_anomalies_count': ('Общее количество обнаруженных аномалий', model_id_label_list, Counter),
            'feature_reconstruction_error_mse': ('MSE ошибка для отдельного признака в последнем окне', feature_error_label_list, Gauge),
            'last_successful_run_timestamp_seconds': ('Timestamp последнего успешного цикла детекции', [], Gauge), 
            'data_points_in_current_window': ('Количество точек данных в текущем анализируемом окне', [], Gauge)
        }
        for name_suffix, (doc, labels, metric_type) in metric_definitions.items():
            full_name = self.metrics_prefix + name_suffix
            # Используем исправленные имена глобальных переменных
            global_var_name = f"PROM_{name_suffix.upper()}" 
            if globals().get(global_var_name) is not None: # Проверяем существование и значение
                if full_name in REGISTRY._names_to_collectors:
                    try: REGISTRY.unregister(REGISTRY._names_to_collectors[full_name]); logging.info(f"Удалена старая метрика: {full_name}")
                    except Exception as e: logging.warning(f"Не удалось удалить старую метрику {full_name}: {e}")
            current_labels = tuple(labels) if labels else ()
            if metric_type == Gauge: globals()[global_var_name] = Gauge(full_name, doc, labelnames=current_labels)
            elif metric_type == Counter: globals()[global_var_name] = Counter(full_name, doc, labelnames=current_labels)
        logging.info("Prometheus метрики инициализированы.")
        if PROM_TOTAL_ANOMALIES_COUNT: # Инициализация счетчиков
            try:
                PROM_TOTAL_ANOMALIES_COUNT.labels(model_id="A").inc(0)
                PROM_TOTAL_ANOMALIES_COUNT.labels(model_id="B").inc(0)
                logging.info("Счетчики total_anomalies_count инициализированы.")
            except Exception as e: logging.warning(f"Не удалось инициализировать счетчики: {e}")


    def _process_model_output(self, model, model_id_label: str, sequence_to_predict: np.ndarray, threshold: float):
        # Используем ИСПРАВЛЕННЫЕ имена глобальных переменных метрик
        if model is None: 
            logging.warning(f"Модель {model_id_label} не загружена.")
            if PROM_LATEST_RECONSTRUCTION_ERROR_MSE: PROM_LATEST_RECONSTRUCTION_ERROR_MSE.labels(model_id=model_id_label).set(0)
            if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.labels(model_id=model_id_label).set(0)
            if PROM_FEATURE_RECONSTRUCTION_ERROR_MSE:
                for fnk in self.metric_columns_ordered: PROM_FEATURE_RECONSTRUCTION_ERROR_MSE.labels(model_id=model_id_label,feature_name=fnk).set(0)
            return
        try:
            reconstructed_sequence = model.predict(sequence_to_predict, verbose=0)
            mse = np.mean(np.power(sequence_to_predict - reconstructed_sequence, 2))
            logging.info(f"[{model_id_label}] Общая MSE: {mse:.6f}")
            if PROM_LATEST_RECONSTRUCTION_ERROR_MSE: PROM_LATEST_RECONSTRUCTION_ERROR_MSE.labels(model_id=model_id_label).set(mse)
            
            sq_err_elems = np.power(sequence_to_predict[0] - reconstructed_sequence[0], 2)
            mse_per_feature = np.mean(sq_err_elems, axis=0)
            if PROM_FEATURE_RECONSTRUCTION_ERROR_MSE:
                for i, fnk in enumerate(self.metric_columns_ordered):
                    try:
                        cfm = mse_per_feature[i]
                        PROM_FEATURE_RECONSTRUCTION_ERROR_MSE.labels(model_id=model_id_label,feature_name=fnk).set(cfm)
                        if mse > threshold*0.5 or cfm > threshold*0.5: logging.info(f"[{model_id_label}] Ошибка '{fnk}': {cfm:.6f}")
                    except Exception as e: logging.error(f"[{model_id_label}] Ошибка уст. метрики для '{fnk}': {e}")
            is_anomaly = mse > threshold
            if is_anomaly:
                logging.warning(f"!!! [{model_id_label}] АНОМАЛИЯ !!! MSE: {mse:.6f} > Порог: {threshold:.6f}")
                if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.labels(model_id=model_id_label).set(1)
                if PROM_TOTAL_ANOMALIES_COUNT: PROM_TOTAL_ANOMALIES_COUNT.labels(model_id=model_id_label).inc()
                fer = [f"[{model_id_label}] Ошибки по признакам (аномалия):"]
                for i, fnk in enumerate(self.metric_columns_ordered): fer.append(f"  - '{fnk}': {mse_per_feature[i]:.6f}")
                logging.warning("\n".join(fer))
            else:
                logging.info(f"[{model_id_label}] Норма. MSE: {mse:.6f} <= Порог: {threshold:.6f}")
                if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.labels(model_id=model_id_label).set(0)
        except Exception as e:
            logging.error(f"[{model_id_label}] Ошибка предсказания: {e}", exc_info=True)
            if PROM_LATEST_RECONSTRUCTION_ERROR_MSE: PROM_LATEST_RECONSTRUCTION_ERROR_MSE.labels(model_id=model_id_label).set(-1)
            if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.labels(model_id=model_id_label).set(0)
            if PROM_FEATURE_RECONSTRUCTION_ERROR_MSE:
                for fnk in self.metric_columns_ordered: PROM_FEATURE_RECONSTRUCTION_ERROR_MSE.labels(model_id=model_id_label,feature_name=fnk).set(-1) 

    def run_detection_cycle(self):
        logging.info("Начало цикла.")
        current_window_df = self._fetch_data_window()
        # Используем ИСПРАВЛЕННЫЕ имена глобальных переменных метрик
        if PROM_DATA_POINTS_IN_CURRENT_WINDOW: PROM_DATA_POINTS_IN_CURRENT_WINDOW.set(len(current_window_df) if current_window_df is not None else 0)
        if current_window_df is None or current_window_df.empty:
            logging.warning("Нет данных для окна. Пропуск.")
            self._process_model_output(None, "A", np.array([]), self.anomaly_threshold_a)
            self._process_model_output(None, "B", np.array([]), self.anomaly_threshold_b)
            return
        sequence_to_predict = self._preprocess_and_create_sequence(current_window_df.copy())
        if sequence_to_predict is None:
            logging.warning("Не удалось предобработать данные. Пропуск.")
            self._process_model_output(None, "A", np.array([]), self.anomaly_threshold_a)
            self._process_model_output(None, "B", np.array([]), self.anomaly_threshold_b)
            return
        logging.info("--- Обработка Моделью A ---")
        self._process_model_output(self.model_a, "A", sequence_to_predict, self.anomaly_threshold_a)
        logging.info("--- Обработка Моделью B ---")
        self._process_model_output(self.model_b, "B", sequence_to_predict, self.anomaly_threshold_b)
        if PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS: PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS.set_to_current_time()
        logging.info("Цикл завершен.")

    def start_server_and_loop(self):
        if not self.scaler or (not self.model_a and not self.model_b) :
             logging.error("Не загружен скейлер или обе модели. Запуск невозможен.")
             exit(1)
        try: start_http_server(self.exporter_port); logging.info(f"Prometheus exporter на порту {self.exporter_port}")
        except OSError as e: 
            if e.errno==98 or e.errno==10048: logging.error(f"Порт {self.exporter_port} занят.")
            else: logging.error(f"OSError exporter: {e}", exc_info=True)
            exit(1) 
        except Exception as e: logging.error(f"Ошибка exporter: {e}", exc_info=True); exit(1)
        while True:
            try: self.run_detection_cycle()
            except Exception as e: logging.error(f"Крит. ошибка в цикле: {e}", exc_info=True)
            logging.info(f"Ожидание {self.query_interval}с...")
            time.sleep(self.query_interval)

if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent / "config.yaml"
    PROM_LATEST_RECONSTRUCTION_ERROR_MSE = None # ИСПРАВЛЕНО
    PROM_IS_ANOMALY_DETECTED = None
    PROM_TOTAL_ANOMALIES_COUNT = None
    PROM_FEATURE_RECONSTRUCTION_ERROR_MSE = None # ИСПРАВЛЕНО
    PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS = None # ИСПРАВЛЕНО
    PROM_DATA_POINTS_IN_CURRENT_WINDOW = None # ИСПРАВЛЕНО
    detector = RealtimeAnomalyDetector(config_path=config_file)
    detector.start_server_and_loop()