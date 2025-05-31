import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Функции ---

def load_config(config_path: Path) -> dict:
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        logging.info(f"Конфигурация успешно загружена из {config_path}")
        return config_data
    except FileNotFoundError:
        logging.error(f"Ошибка: Файл конфигурации не найден по пути {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Ошибка парсинга YAML файла {config_path}: {e}")
        exit(1)
    return {}

def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Загружает предобработанные данные из Parquet файла."""
    if not file_path.exists():
        logging.error(f"Ошибка: Файл данных не найден по пути {file_path}")
        exit(1)
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Предобработанные данные успешно загружены из {file_path}. Размер: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных из {file_path}: {e}")
        exit(1)

def load_tf_model(model_path: Path):
    """Загружает обученную модель TensorFlow/Keras."""
    if not model_path.exists():
        logging.error(f"Файл модели не найден: {model_path}")
        exit(1)
    try:
        model = load_model(model_path) # compile=False можно добавить, если не планируется дообучение
        logging.info(f"Модель успешно загружена из {model_path}")
        model.summary(print_fn=logging.info)
        return model
    except Exception as e:
        logging.error(f"Ошибка загрузки модели из {model_path}: {e}", exc_info=True)
        exit(1)

def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """Создает последовательности (окна) из временного ряда."""
    xs = []
    if len(data) < sequence_length:
        logging.warning(f"Данных ({len(data)}) меньше, чем длина последовательности ({sequence_length}). Невозможно создать последовательности.")
        return np.array(xs)
        
    for i in range(len(data) - sequence_length + 1):
        x = data[i:(i + sequence_length)]
        xs.append(x)
    return np.array(xs)

# --- Основной блок ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    
    CONFIG = load_config(CONFIG_FILE_PATH)

    # Извлечение необходимых настроек из разных секций
    preprocess_settings = CONFIG.get('preprocessing_settings', {})
    training_settings = CONFIG.get('training_settings', {})
    rt_detection_settings = CONFIG.get('real_time_anomaly_detection', {})
    filtering_settings = CONFIG.get('data_filtering_settings', {})

    input_processed_filename = preprocess_settings.get('processed_output_filename')
    model_filename = training_settings.get('model_output_filename')
    sequence_length = training_settings.get('sequence_length')
    anomaly_threshold_mse = rt_detection_settings.get('anomaly_threshold_mse')

    if not all([input_processed_filename, model_filename, sequence_length, anomaly_threshold_mse]):
        logging.error("Не все необходимые параметры (processed_output_filename, model_output_filename, "
                      "sequence_length, anomaly_threshold_mse) найдены в config.yaml. Проверьте секции "
                      "preprocessing_settings, training_settings, real_time_anomaly_detection.")
        exit(1)
    
    input_file_path = BASE_DIR / input_processed_filename
    model_path = BASE_DIR / model_filename

    # Имена выходных файлов
    normal_seq_output_filename = filtering_settings.get('normal_sequences_output_filename', 'filtered_normal_sequences.npy')
    anomalous_seq_output_filename = filtering_settings.get('anomalous_sequences_output_filename', 'filtered_anomalous_sequences.npy')
    # all_errors_output_filename = filtering_settings.get('all_sequence_errors_output_filename') # Если нужно сохранять все ошибки

    normal_seq_output_path = BASE_DIR / normal_seq_output_filename
    anomalous_seq_output_path = BASE_DIR / anomalous_seq_output_filename
    # if all_errors_output_filename:
    #     all_errors_output_path = BASE_DIR / all_errors_output_filename

    logging.info("--- Начало скрипта фильтрации аномальных данных ---")

    # 1. Загрузка предобработанных данных (масштабированных)
    df_processed = load_processed_data(input_file_path)
    if df_processed.empty:
        logging.error("Загруженный DataFrame пуст. Фильтрация невозможна.")
        exit(1)
    
    data_values = df_processed.values # NumPy массив

    # 2. Загрузка обученной модели
    model = load_tf_model(model_path)

    # 3. Создание последовательностей
    all_sequences = create_sequences(data_values, sequence_length)
    if all_sequences.shape[0] == 0:
        logging.error("Не удалось создать ни одной последовательности из данных. Проверьте длину данных и sequence_length.")
        exit(1)
    logging.info(f"Создано {all_sequences.shape[0]} последовательностей для анализа.")

    # 4. Получение реконструкций и расчет ошибок MSE для каждой последовательности
    logging.info("Получение реконструкций от модели...")
    reconstructed_sequences = model.predict(all_sequences, batch_size=training_settings.get('batch_size', 64)) # Используем batch_size из обучения для эффективности
    
    # Расчет MSE для каждой последовательности
    # all_sequences и reconstructed_sequences имеют форму (num_samples, sequence_length, num_features)
    ms_errors = np.mean(np.power(all_sequences - reconstructed_sequences, 2), axis=(1, 2))
    logging.info(f"Рассчитаны ошибки реконструкции для {len(ms_errors)} последовательностей.")

    # if all_errors_output_filename:
    #     try:
    #         np.save(all_errors_output_path, ms_errors)
    #         logging.info(f"Все ошибки реконструкции сохранены в: {all_errors_output_path}")
    #     except Exception as e:
    #         logging.error(f"Ошибка при сохранении всех ошибок реконструкции: {e}")


    # 5. Фильтрация последовательностей на основе порога
    normal_mask = ms_errors <= anomaly_threshold_mse
    anomalous_mask = ms_errors > anomaly_threshold_mse

    normal_sequences = all_sequences[normal_mask]
    anomalous_sequences = all_sequences[anomalous_mask]

    num_normal = normal_sequences.shape[0]
    num_anomalous = anomalous_sequences.shape[0]
    total_sequences = all_sequences.shape[0]

    logging.info(f"--- Результаты фильтрации ---")
    logging.info(f"Использованный порог MSE: {anomaly_threshold_mse:.6f}")
    logging.info(f"Всего последовательностей: {total_sequences}")
    logging.info(f"Нормальных последовательностей: {num_normal} ({num_normal/total_sequences:.2%})")
    logging.info(f"Аномальных последовательностей: {num_anomalous} ({num_anomalous/total_sequences:.2%})")

    # 6. Сохранение отфильтрованных последовательностей
    try:
        np.save(normal_seq_output_path, normal_sequences)
        logging.info(f"Нормальные последовательности сохранены в: {normal_seq_output_path} (формат: {normal_sequences.shape})")
    except Exception as e:
        logging.error(f"Ошибка при сохранении нормальных последовательностей: {e}")

    try:
        np.save(anomalous_seq_output_path, anomalous_sequences)
        logging.info(f"Аномальные последовательности сохранены в: {anomalous_seq_output_path} (формат: {anomalous_sequences.shape})")
    except Exception as e:
        logging.error(f"Ошибка при сохранении аномальных последовательностей: {e}")

    logging.info("--- Скрипт фильтрации аномальных данных завершен ---")