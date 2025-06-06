import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml # Для чтения YAML
from pathlib import Path # Для удобной работы с путями

# --- Глобальные переменные для конфигурации (будут загружены из файла) ---
CONFIG = {}

# --- Функции ---

def load_config(config_path: Path) -> dict:
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        print(f"Конфигурация успешно загружена из {config_path}")
        return config_data
    except FileNotFoundError:
        print(f"Ошибка: Файл конфигурации не найден по пути {config_path}")
        exit(1) # Выход из программы, если конфиг не найден
    except yaml.YAMLError as e:
        print(f"Ошибка парсинга YAML файла {config_path}: {e}")
        exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при загрузке конфигурации {config_path}: {e}")
        exit(1)


def query_prometheus_range(prometheus_url: str, query: str, start_time: datetime, end_time: datetime, step: str) -> pd.DataFrame:
    api_url = f"{prometheus_url}/api/v1/query_range"
    params = {
        'query': query,
        'start': start_time.timestamp(),
        'end': end_time.timestamp(),
        'step': step
    }
    print(f"Запрос к Prometheus: {query} за период с {start_time} по {end_time} с шагом {step}")
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к Prometheus: {e}")
        return pd.DataFrame()
    except ValueError as e: # или requests.exceptions.JSONDecodeError
        print(f"Ошибка декодирования JSON ответа от Prometheus: {e}")
        print(f"Текст ответа: {response.text}")
        return pd.DataFrame()

    if data['status'] == 'success':
        all_series_data = []
        for result in data['data']['result']:
            metric_labels = result.get('metric', {})
            values = result.get('values', [])
            if not values:
                print(f"Предупреждение: для метрики {metric_labels} не вернулось данных.")
                continue
            
            df_series = pd.DataFrame(values, columns=['timestamp', 'value'])
            df_series['timestamp'] = pd.to_datetime(df_series['timestamp'], unit='s')
            df_series['value'] = pd.to_numeric(df_series['value'], errors='coerce')
            df_series = df_series.set_index('timestamp')
            all_series_data.append(df_series)

        if not all_series_data:
            print("Данные не найдены для запроса.")
            return pd.DataFrame()
        
        if len(all_series_data) == 1:
             return all_series_data[0]
        elif len(all_series_data) > 1:
            print(f"Предупреждение: запрос '{query}' вернул {len(all_series_data)} временных рядов. Возвращаем первый.")
            return all_series_data[0] 
        else:
            return pd.DataFrame()
    else:
        print(f"Ошибка в статусе ответа Prometheus: {data.get('errorType')}, {data.get('error')}")
        return pd.DataFrame()

def collect_training_data(prometheus_url: str, queries_dict: dict, start_time: datetime, end_time: datetime, step: str) -> pd.DataFrame:
    all_data_frames = []
    expected_column_names = list(queries_dict.keys())

    for custom_name, query_string in queries_dict.items():
        df_metric = query_prometheus_range(prometheus_url, query_string, start_time, end_time, step)
        if not df_metric.empty:
            df_metric = df_metric.rename(columns={'value': custom_name})
            all_data_frames.append(df_metric)
        else:
            print(f"Не удалось получить данные для '{custom_name}' ({query_string})")

    if not all_data_frames:
        print("Не удалось собрать никаких данных.")
        return pd.DataFrame()

    final_df = pd.concat(all_data_frames, axis=1, join='outer')
    
    for col_name in expected_column_names:
        if col_name in final_df.columns:
            if final_df[col_name].dtype == 'object':
                 final_df[col_name] = pd.to_numeric(final_df[col_name], errors='coerce')
        else:
            # Если колонка ожидалась, но не была получена, добавляем ее с NA значениями
            final_df[col_name] = pd.NA 

    final_df = final_df.sort_index()

    # Добавляем временные признаки для дня недели и часа суток
    final_df['day_of_week'] = final_df.index.dayofweek.astype(int)
    final_df['hour_of_day'] = final_df.index.hour.astype(int)

    return final_df

# --- Основной блок ---
if __name__ == "__main__":
    # Определяем путь к файлу конфигурации относительно текущего скрипта
    CONFIG_FILE_PATH = Path(__file__).parent / "config.yaml"
    CONFIG = load_config(CONFIG_FILE_PATH)

    PROMETHEUS_URL = CONFIG.get('prometheus_url')
    QUERIES = CONFIG.get('queries')
    DATA_SETTINGS = CONFIG.get('data_settings', {}) # Получаем вложенный словарь, или пустой, если его нет

    if not PROMETHEUS_URL or not QUERIES:
        print("Ошибка: 'prometheus_url' или 'queries' не найдены в файле конфигурации.")
        exit(1)

    # Определение временного диапазона
    collection_period_hours = DATA_SETTINGS.get('collection_period_hours')
    start_time_iso = DATA_SETTINGS.get('start_time_iso')
    end_time_iso = DATA_SETTINGS.get('end_time_iso')

    if start_time_iso and end_time_iso and (not collection_period_hours or collection_period_hours == 0) :
        try:
            START_TIME = datetime.fromisoformat(start_time_iso)
            END_TIME = datetime.fromisoformat(end_time_iso)
            if START_TIME >= END_TIME:
                print("Ошибка: 'start_time_iso' должен быть раньше 'end_time_iso'.")
                exit(1)
        except ValueError:
            print("Ошибка: 'start_time_iso' или 'end_time_iso' имеют неверный формат. Используйте YYYY-MM-DDTHH:MM:SS.")
            exit(1)
    elif collection_period_hours and collection_period_hours > 0:
        END_TIME = datetime.now()
        START_TIME = END_TIME - timedelta(hours=collection_period_hours)
    else:
        print("Ошибка: Некорректные настройки времени. Укажите 'collection_period_hours' > 0 или 'start_time_iso' и 'end_time_iso'.")
        exit(1)

    STEP = DATA_SETTINGS.get('step', '30s') # Значение по умолчанию, если не указано
    PARQUET_FILENAME = DATA_SETTINGS.get('output_filename', 'prometheus_metrics_data.parquet')

    print(f"Сбор данных из Prometheus: {PROMETHEUS_URL}")
    print(f"Запросы: {QUERIES}")
    print(f"Период: с {START_TIME.strftime('%Y-%m-%d %H:%M:%S')} по {END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Шаг: {STEP}")
    print(f"Файл для сохранения: {PARQUET_FILENAME}")
    print("-" * 30)

    training_data = collect_training_data(PROMETHEUS_URL, QUERIES, START_TIME, END_TIME, STEP)

    if not training_data.empty:
        print("\n--- Собранные данные (первые 5 и последние 5 строк) ---")
        print(training_data.head())
        print("...")
        print(training_data.tail())
        print(f"\nРазмер DataFrame: {training_data.shape}")
        print("\nИнформация о DataFrame (до сохранения):")
        training_data.info()

        try:
            output_file_path = Path(__file__).parent / PARQUET_FILENAME
            training_data.to_parquet(output_file_path, engine='pyarrow', index=True)
            print(f"\nДанные успешно сохранены в {output_file_path}")
        except Exception as e:
            print(f"\nОшибка при сохранении данных в Parquet: {e}")
            print("Убедитесь, что библиотека 'pyarrow' (или 'fastparquet') установлена.")
    else:
        print("\nНе удалось собрать данные для обучения.")