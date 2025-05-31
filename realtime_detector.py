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

# --- Глобальные переменные для Prometheus метрик ---
# Инициализируем их как None, чтобы можно было определить, настроены ли они
PROM_LATEST_RECONSTRUCTION_ERROR = None
PROM_IS_ANOMALY_DETECTED = None
PROM_TOTAL_ANOMALIES_COUNT = None
PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP = None
PROM_DATA_POINTS_IN_WINDOW = None
PROM_FEATURE_RECONSTRUCTION_ERROR = None # Новая метрика


class RealtimeAnomalyDetector:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Извлечение настроек
        self.prom_url = self.config.get('prometheus_url')
        self.queries = self.config.get('queries', {})
        # self.metric_columns_ordered будет содержать ключи из self.queries,
        # которые мы используем как feature_name в Prometheus метках
        self.metric_columns_ordered = list(self.queries.keys()) 

        rt_config = self.config.get('real_time_anomaly_detection', {})
        self.query_interval = rt_config.get('query_interval_seconds', 60)
        self.anomaly_threshold = rt_config.get('anomaly_threshold_mse', 0.0025) # ЗАГЛУШКА!
        self.exporter_port = rt_config.get('exporter_port', 8001)
        self.metrics_prefix = rt_config.get('metrics_prefix', 'anomaly_detector_')

        # Используем scaler_output_filename и model_output_filename из соответствующих секций
        preprocess_config = self.config.get('preprocessing_settings', {})
        training_config = self.config.get('training_settings', {})
        
        scaler_filename = preprocess_config.get('scaler_output_filename', 'fitted_scaler.joblib')
        model_filename = training_config.get('model_output_filename', 'lstm_autoencoder_model.keras')
        self.sequence_length = training_config.get('sequence_length', 20)

        # Определяем длительность шага для запроса данных
        data_s_config = self.config.get('data_settings',{})
        self.data_step_duration_str = rt_config.get('data_step_duration', data_s_config.get('step', '30s'))


        # Пути к файлам модели и скейлера
        base_dir = Path(__file__).resolve().parent
        self.scaler_path = base_dir / scaler_filename
        self.model_path = base_dir / model_filename

        self.scaler = self._load_scaler()
        self.model = self._load_tf_model()
        
        if self.model and self.scaler:
            # Ожидаемое количество признаков из модели (или скейлера)
            self.num_features = self.scaler.n_features_in_ # Для скейлера (надежнее, если модель сложная)
            if self.num_features != len(self.queries):
                logging.error(f"Расхождение в количестве признаков! "
                              f"Ожидалось по запросам: {len(self.queries)}, "
                              f"получено из скейлера: {self.num_features}. Проверьте config.yaml (queries) и обученный скейлер.")
                exit(1)

        self._setup_prometheus_metrics()
        # self.data_buffer = pd.DataFrame() # Пока не используется активно


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
            exit(1)
        try:
            scaler = joblib.load(self.scaler_path)
            logging.info(f"Скейлер успешно загружен из {self.scaler_path}")
            return scaler
        except Exception as e:
            logging.error(f"Ошибка загрузки скейлера из {self.scaler_path}: {e}", exc_info=True)
            exit(1)

    def _load_tf_model(self):
        if not self.model_path.exists():
            logging.error(f"Файл модели не найден: {self.model_path}")
            exit(1)
        try:
            # Загрузка модели с выключенной компиляцией для инференса может быть быстрее
            # model = load_model(self.model_path, compile=False) 
            model = load_model(self.model_path)
            model.summary(print_fn=logging.info) # Выводим summary в лог
            logging.info(f"Модель успешно загружена из {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"Ошибка загрузки модели из {self.model_path}: {e}", exc_info=True)
            exit(1)

    def _td_seconds(self, td_str: str) -> int:
        """Преобразует строку типа '30s', '1m', '1h' в секунды."""
        if td_str.endswith('s'):
            return int(td_str[:-1])
        elif td_str.endswith('m'):
            return int(td_str[:-1]) * 60
        elif td_str.endswith('h'):
            return int(td_str[:-1]) * 3600
        try: # Попробуем просто как число секунд
            return int(td_str)
        except ValueError:
            logging.warning(f"Не удалось распознать длительность шага: {td_str}. Используется 30с по умолчанию.")
            return 30

    def _fetch_data_window(self) -> pd.DataFrame | None:
        """Запрашивает окно данных из Prometheus, достаточное для одной последовательности."""
        step_seconds = self._td_seconds(self.data_step_duration_str)
        # Длительность окна данных для одной последовательности
        window_duration_seconds = self.sequence_length * step_seconds
        
        end_time = datetime.now()
        # Запрашиваем данные с небольшим запасом, чтобы гарантированно получить sequence_length точек
        # с учетом возможных округлений времени от Prometheus и выравнивания шага.
        start_time_query = end_time - timedelta(seconds=window_duration_seconds + step_seconds * 2)


        all_metric_dfs = []
        logging.info(f"Запрос данных с {start_time_query.strftime('%Y-%m-%d %H:%M:%S')} по {end_time.strftime('%Y-%m-%d %H:%M:%S')} с шагом {self.data_step_duration_str}")

        for custom_name in self.metric_columns_ordered: # Итерируемся в нужном порядке
            query_string = self.queries[custom_name]
            api_url = f"{self.prom_url}/api/v1/query_range"
            params = {
                'query': query_string,
                'start': start_time_query.timestamp(),
                'end': end_time.timestamp(),
                'step': self.data_step_duration_str 
            }
            try:
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status() # Проверка на HTTP ошибки (4xx, 5xx)
                data = response.json()
                
                if data['status'] == 'success' and data['data']['result']:
                    values = data['data']['result'][0].get('values', [])
                    if values:
                        df_metric = pd.DataFrame(values, columns=['timestamp', custom_name])
                        df_metric['timestamp'] = pd.to_datetime(df_metric['timestamp'], unit='s')
                        df_metric[custom_name] = pd.to_numeric(df_metric[custom_name], errors='coerce')
                        df_metric = df_metric.set_index('timestamp')
                        all_metric_dfs.append(df_metric)
                    else:
                        logging.warning(f"Нет данных для метрики '{custom_name}' в ответе Prometheus.")
                        # Если хотя бы одна метрика не пришла, мы не можем сформировать полное окно
                        return None 
                else:
                    logging.warning(f"Не удалось получить данные или пустой результат для '{custom_name}': {data.get('errorType', '')} {data.get('error', data.get('status'))}")
                    return None # Аналогично, если проблема с одной из метрик
            except requests.exceptions.RequestException as e:
                logging.error(f"Ошибка запроса к Prometheus для '{custom_name}': {e}")
                return None
            except Exception as e: # Включая JSONDecodeError
                logging.error(f"Неожиданная ошибка при получении или обработке данных для '{custom_name}': {e}", exc_info=True)
                return None

        if not all_metric_dfs or len(all_metric_dfs) != len(self.queries):
            logging.warning("Не все метрики были успешно загружены или не содержали данных.")
            return None

        # Объединяем DataFrame. inner join чтобы были только общие таймстемпы
        # Это может привести к потере данных, если у метрик разное время.
        # Альтернатива - outer join и затем ffill/bfill, но это усложнит логику.
        # Для стабильной системы inner join должен работать.
        try:
            final_df = pd.concat(all_metric_dfs, axis=1, join='inner') 
        except Exception as e:
            logging.error(f"Ошибка при объединении DataFrame метрик: {e}", exc_info=True)
            return None
            
        if final_df.empty:
            logging.warning("Итоговый DataFrame после объединения метрик пуст (нет общих временных меток).")
            return None

        # Приводим к порядку колонок как при обучении и проверяем наличие всех
        try:
            final_df = final_df[self.metric_columns_ordered]
        except KeyError as e:
            logging.error(f"Ошибка приведения колонок к нужному порядку: {e}. "
                          f"Ожидаемые колонки: {self.metric_columns_ordered}. "
                          f"Полученные колонки: {final_df.columns.tolist()}")
            return None


        # Отбираем последние sequence_length точек
        if len(final_df) >= self.sequence_length:
            # Берем самые свежие данные, чтобы избежать старых значений, если Prometheus вернул больше
            return final_df.tail(self.sequence_length)
        else:
            logging.warning(f"Недостаточно данных для формирования полной последовательности после объединения и фильтрации. "
                            f"Получено {len(final_df)} точек, нужно {self.sequence_length}.")
            if PROM_DATA_POINTS_IN_WINDOW: PROM_DATA_POINTS_IN_WINDOW.set(len(final_df))
            return None


    def _preprocess_and_create_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """Предобрабатывает DataFrame и создает одну последовательность."""
        if df.isnull().values.any():
            logging.warning("Обнаружены NaN в полученных данных для окна, применяем ffill().bfill()")
            # Перед ffill/bfill нужно убедиться, что порядок правильный (по индексу времени)
            df = df.sort_index() 
            df = df.ffill().bfill() # Заполняем пропуски
            if df.isnull().values.any():
                logging.error("NaN остались даже после ffill/bfill. Пропуск этого окна.")
                # Можно логировать, какие именно колонки содержат NaN
                nan_cols = df.columns[df.isnull().any()].tolist()
                logging.error(f"Колонки с оставшимися NaN: {nan_cols}")
                return None
        
        try:
            # Масштабирование данных
            # Убедимся, что данные в правильном порядке колонок перед .values
            scaled_values = self.scaler.transform(df[self.metric_columns_ordered].values)
        except Exception as e:
            logging.error(f"Ошибка при масштабировании данных: {e}", exc_info=True)
            logging.error(f"Данные, вызвавшие ошибку (форма {df[self.metric_columns_ordered].shape}, первые 2 строки):\n{df[self.metric_columns_ordered].head(2)}")
            logging.error(f"Ожидаемое количество признаков скейлером: {self.scaler.n_features_in_}")
            logging.error(f"Фактическое количество признаков в данных: {df[self.metric_columns_ordered].shape[1]}")
            return None

        # Создание последовательности (батч из 1 элемента, sequence_length, num_features)
        sequence = np.expand_dims(scaled_values, axis=0)
        return sequence

    def _setup_prometheus_metrics(self):
        """Инициализирует Prometheus метрики."""
        global PROM_LATEST_RECONSTRUCTION_ERROR, PROM_IS_ANOMALY_DETECTED, \
               PROM_TOTAL_ANOMALIES_COUNT, PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP, \
               PROM_DATA_POINTS_IN_WINDOW, PROM_FEATURE_RECONSTRUCTION_ERROR
        
        # Список имен метрик для очистки при перезапуске
        metric_names_to_clear = [
            self.metrics_prefix + 'latest_reconstruction_error_mse',
            self.metrics_prefix + 'is_anomaly_detected',
            self.metrics_prefix + 'total_anomalies_count',
            self.metrics_prefix + 'last_successful_run_timestamp_seconds',
            self.metrics_prefix + 'data_points_in_current_window',
            self.metrics_prefix + 'feature_reconstruction_error_mse' # Имя новой метрики
        ]
        for name in metric_names_to_clear:
            # REGISTRY._collector_to_names - это внутреннее API, но оно полезно для проверки
            # Более безопасный способ - просто пытаться unregister и ловить KeyError
            if name in REGISTRY._names_to_collectors: # Проверяем, существует ли метрика
                try:
                    REGISTRY.unregister(REGISTRY._names_to_collectors[name])
                    logging.info(f"Удалена старая метрика: {name}")
                except KeyError: # pragma: no cover
                    # Это может произойти, если метрика была удалена другим способом или в гонке состояний
                    logging.debug(f"Метрика {name} уже была удалена или не найдена для unregister.")
                except Exception as e: # pragma: no cover
                    logging.warning(f"Не удалось удалить старую метрику {name}: {e}")


        PROM_LATEST_RECONSTRUCTION_ERROR = Gauge(
            self.metrics_prefix + 'latest_reconstruction_error_mse',
            'MSE ошибка реконструкции для последнего проанализированного окна'
        )
        PROM_IS_ANOMALY_DETECTED = Gauge(
            self.metrics_prefix + 'is_anomaly_detected',
            'Флаг аномалии (1 если аномалия, 0 если норма)'
        )
        PROM_TOTAL_ANOMALIES_COUNT = Counter(
            self.metrics_prefix + 'total_anomalies_count',
            'Общее количество обнаруженных аномалий с момента запуска'
        )
        PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP = Gauge(
            self.metrics_prefix + 'last_successful_run_timestamp_seconds',
            'Timestamp последнего успешного цикла детекции'
        )
        PROM_DATA_POINTS_IN_WINDOW = Gauge(
            self.metrics_prefix + 'data_points_in_current_window',
            'Количество точек данных в текущем анализируемом окне'
        )
        # Новая метрика для ошибок по каждому признаку
        PROM_FEATURE_RECONSTRUCTION_ERROR = Gauge(
            self.metrics_prefix + 'feature_reconstruction_error_mse',
            'MSE ошибка реконструкции для отдельного признака в последнем окне',
            ['feature_name']  # Метка для имени признака
        )
        logging.info("Prometheus метрики инициализированы.")


    def run_detection_cycle(self):
        """Выполняет один цикл получения данных, обработки и детекции аномалий."""
        logging.info("Начало нового цикла детекции...")
        
        current_window_df = self._fetch_data_window()

        # Обработка случая, если данные не получены
        if current_window_df is None or current_window_df.empty:
            logging.warning("Не удалось получить или подготовить данные для текущего окна. Пропуск цикла.")
            if PROM_LATEST_RECONSTRUCTION_ERROR: PROM_LATEST_RECONSTRUCTION_ERROR.set(0) 
            if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.set(0) # Считаем, что нет аномалии, если нет данных
            if PROM_DATA_POINTS_IN_WINDOW: PROM_DATA_POINTS_IN_WINDOW.set(0)
            # Для ошибок по признакам: устанавливаем в 0 для всех известных признаков
            if PROM_FEATURE_RECONSTRUCTION_ERROR:
                for feature_name_key in self.metric_columns_ordered:
                    try:
                        PROM_FEATURE_RECONSTRUCTION_ERROR.labels(feature_name=feature_name_key).set(0)
                    except Exception as e: # pragma: no cover
                        logging.error(f"Ошибка установки метрики для {feature_name_key} при отсутствии данных: {e}")
            return

        if PROM_DATA_POINTS_IN_WINDOW: PROM_DATA_POINTS_IN_WINDOW.set(len(current_window_df))
        
        # Предобработка и создание последовательности
        sequence_to_predict = self._preprocess_and_create_sequence(current_window_df.copy()) # copy()

        if sequence_to_predict is None:
            logging.warning("Не удалось предобработать данные или создать последовательность. Пропуск цикла.")
            if PROM_LATEST_RECONSTRUCTION_ERROR: PROM_LATEST_RECONSTRUCTION_ERROR.set(0)
            if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.set(0)
            if PROM_FEATURE_RECONSTRUCTION_ERROR:
                for feature_name_key in self.metric_columns_ordered:
                    try:
                        PROM_FEATURE_RECONSTRUCTION_ERROR.labels(feature_name=feature_name_key).set(0)
                    except Exception as e: # pragma: no cover
                         logging.error(f"Ошибка установки метрики для {feature_name_key} при ошибке предобработки: {e}")
            return
            
        # Предсказание и расчет ошибки
        try:
            reconstructed_sequence = self.model.predict(sequence_to_predict, verbose=0)
            mse = np.mean(np.power(sequence_to_predict - reconstructed_sequence, 2)) # Общая MSE
            logging.info(f"Общая ошибка реконструкции (MSE): {mse:.6f}")
            if PROM_LATEST_RECONSTRUCTION_ERROR: PROM_LATEST_RECONSTRUCTION_ERROR.set(mse)

            # --- Расчет и публикация ошибок по признакам ---
            squared_errors_per_element = np.power(sequence_to_predict[0] - reconstructed_sequence[0], 2)
            mse_per_feature = np.mean(squared_errors_per_element, axis=0) # (num_features,)
            
            if PROM_FEATURE_RECONSTRUCTION_ERROR:
                for i, feature_name_key in enumerate(self.metric_columns_ordered):
                    try:
                        PROM_FEATURE_RECONSTRUCTION_ERROR.labels(feature_name=feature_name_key).set(mse_per_feature[i])
                        # Логирование ошибок по признакам можно сделать более редким или по условию
                        if mse > self.anomaly_threshold * 0.5: # Логируем если ошибка значительная
                             logging.info(f"Ошибка для '{feature_name_key}': {mse_per_feature[i]:.6f}")
                    except IndexError: # pragma: no cover
                        logging.error(f"IndexError при доступе к mse_per_feature[{i}] для '{feature_name_key}'. Длина mse_per_feature: {len(mse_per_feature)}")
                    except Exception as e: # pragma: no cover
                        logging.error(f"Ошибка установки метрики для {feature_name_key}: {e}")

            # --- Конец блока ошибок по признакам ---

        except Exception as e:
            logging.error(f"Ошибка во время предсказания моделью: {e}", exc_info=True)
            if PROM_LATEST_RECONSTRUCTION_ERROR: PROM_LATEST_RECONSTRUCTION_ERROR.set(-1) # Ошибка предсказания
            if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.set(0) # Не можем сказать
            if PROM_FEATURE_RECONSTRUCTION_ERROR:
                for feature_name_key in self.metric_columns_ordered: 
                    try:
                        PROM_FEATURE_RECONSTRUCTION_ERROR.labels(feature_name=feature_name_key).set(-1) 
                    except Exception as e_label: # pragma: no cover
                        logging.error(f"Ошибка установки метрики -1 для {feature_name_key}: {e_label}")
            return

        # Определение аномалии
        is_anomaly = mse > self.anomaly_threshold
        if is_anomaly:
            logging.warning(f"!!! ОБНАРУЖЕНА АНОМАЛИЯ !!! Общая MSE: {mse:.6f} > Порог: {self.anomaly_threshold:.6f}")
            if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.set(1)
            if PROM_TOTAL_ANOMALIES_COUNT: PROM_TOTAL_ANOMALIES_COUNT.inc()

            # Логируем детальный отчет по ошибкам признаков только при аномалии
            feature_error_report = ["Ошибки реконструкции по признакам для аномальной последовательности:"]
            for i, feature_name_key in enumerate(self.metric_columns_ordered):
                try:
                    feature_error_report.append(f"  - Метрика '{feature_name_key}': MSE = {mse_per_feature[i]:.6f}")
                except IndexError: # pragma: no cover
                     feature_error_report.append(f"  - Метрика '{feature_name_key}': Ошибка доступа к mse_per_feature")
            logging.warning("\n".join(feature_error_report))
        else:
            logging.info(f"Состояние нормальное. Общая MSE: {mse:.6f} <= Порог: {self.anomaly_threshold:.6f}")
            if PROM_IS_ANOMALY_DETECTED: PROM_IS_ANOMALY_DETECTED.set(0)
            
        if PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP: PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP.set_to_current_time()
        logging.info("Цикл детекции завершен.")

    def start_server_and_loop(self):
        """Запускает HTTP сервер для Prometheus и основной цикл детекции."""
        try:
            start_http_server(self.exporter_port)
            logging.info(f"Prometheus exporter запущен на порту {self.exporter_port}")
        except OSError as e: # Более конкретная ошибка для занятого порта
            if e.errno == 98 or e.errno == 10048 : # Address already in use (Linux/Windows)
                 logging.error(f"Не удалось запустить Prometheus exporter: Порт {self.exporter_port} уже используется.")
            else: # pragma: no cover
                 logging.error(f"Не удалось запустить Prometheus exporter (OSError): {e}", exc_info=True)
            exit(1) # Выход, если сервер не может запуститься
        except Exception as e: # pragma: no cover
            logging.error(f"Не удалось запустить Prometheus exporter: {e}", exc_info=True)
            exit(1)

        # Бесконечный цикл с заданной периодичностью
        while True:
            try:
                self.run_detection_cycle()
            except Exception as e: # pragma: no cover
                # Ловим неожиданные ошибки в основном цикле, чтобы он не падал
                logging.error(f"Критическая ошибка в run_detection_cycle: {e}", exc_info=True)
                # Можно добавить логику для попытки перезапуска или безопасной остановки
            
            logging.info(f"Ожидание {self.query_interval} секунд перед следующим циклом...")
            time.sleep(self.query_interval)

# --- Основной блок ---
if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent / "config.yaml"
    
    # Установим глобальные переменные метрик в None перед созданием детектора,
    # чтобы _setup_prometheus_metrics корректно их инициализировал или пересоздал.
    # Это важно, если скрипт может быть перезапущен в среде, где объект REGISTRY сохраняется.
    PROM_LATEST_RECONSTRUCTION_ERROR = None
    PROM_IS_ANOMALY_DETECTED = None
    PROM_TOTAL_ANOMALIES_COUNT = None
    PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP = None
    PROM_DATA_POINTS_IN_WINDOW = None
    PROM_FEATURE_RECONSTRUCTION_ERROR = None

    detector = RealtimeAnomalyDetector(config_path=config_file)
    detector.start_server_and_loop()