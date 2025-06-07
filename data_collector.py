import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml
from pathlib import Path
from diskcache import Cache

# --- Глобальные переменные ---
CONFIG = {}
CACHE: Cache | None = None

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
        exit(1)
    except yaml.YAMLError as e:
        print(f"Ошибка парсинга YAML файла {config_path}: {e}")
        exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при загрузке конфигурации {config_path}: {e}")
        exit(1)


def query_prometheus_range(prometheus_url: str, query: str, start_time: datetime, end_time: datetime, step: str) -> pd.DataFrame:
    """Запрашивает данные из Prometheus, используя кеш для ускорения повторных запросов."""
    if CACHE is not None:
        cache_key = (prometheus_url, query, start_time.isoformat(), end_time.isoformat(), step)
        cached_result = CACHE.get(cache_key)
        if cached_result is not None:
            # Убедимся, что загружаем копию, чтобы избежать проблем с изменяемостью
            return cached_result.copy()

    api_url = f"{prometheus_url}/api/v1/query_range"
    params = {'query': query, 'start': start_time.timestamp(), 'end': end_time.timestamp(), 'step': step}
    print(f"  CACHE MISS: Запрос к Prometheus: {query[:70]}... ({start_time.strftime('%Y-%m-%d %H:%M')} -> {end_time.strftime('%Y-%m-%d %H:%M')})")
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"    -> Ошибка при запросе к Prometheus: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"    -> Ошибка декодирования JSON ответа от Prometheus: {e}\n    Текст ответа: {response.text}")
        return pd.DataFrame()

    result_df = pd.DataFrame()
    if data['status'] == 'success':
        all_series_data = []
        for result in data['data']['result']:
            metric_labels = result.get('metric', {})
            values = result.get('values', [])
            if not values: continue
            
            df_series = pd.DataFrame(values, columns=['timestamp', 'value'])
            df_series['timestamp'] = pd.to_datetime(df_series['timestamp'], unit='s')
            df_series['value'] = pd.to_numeric(df_series['value'], errors='coerce')
            df_series = df_series.set_index('timestamp')
            all_series_data.append(df_series)

        if not all_series_data: pass
        elif len(all_series_data) == 1:
            result_df = all_series_data[0]
        else:
            print(f"    -> Предупреждение: запрос '{query}' вернул {len(all_series_data)} временных рядов. Возвращаем первый.")
            result_df = all_series_data[0]
    else:
        print(f"    -> Ошибка в статусе ответа Prometheus: {data.get('errorType')}, {data.get('error')}")

    if CACHE is not None and not result_df.empty:
        CACHE.set(cache_key, result_df)

    return result_df


def collect_training_data(prometheus_url: str, queries_dict: dict, start_time: datetime, end_time: datetime, step: str, chunk_hours: int) -> pd.DataFrame:
    """
    Собирает данные, разбивая большой временной диапазон на более мелкие фрагменты (чанки) для эффективного кеширования.
    """
    all_chunks_data = []
    current_start = start_time
    print(f"Разбиваем период с {start_time.strftime('%Y-%m-%d %H:%M')} по {end_time.strftime('%Y-%m-%d %H:%M')} на чанки по {chunk_hours} час(а).")

    while current_start < end_time:
        current_end = current_start + timedelta(hours=chunk_hours)
        if current_end > end_time:
            current_end = end_time

        print(f"\n-- Сбор данных для чанка: {current_start.strftime('%Y-%m-%d %H:%M')} -> {current_end.strftime('%Y-%m-%d %H:%M')}")
        
        metrics_for_chunk = []
        for custom_name, query_string in queries_dict.items():
            df_metric = query_prometheus_range(prometheus_url, query_string, current_start, current_end, step)
            if not df_metric.empty:
                df_metric = df_metric.rename(columns={'value': custom_name})
                metrics_for_chunk.append(df_metric)

        if metrics_for_chunk:
            chunk_df = pd.concat(metrics_for_chunk, axis=1, join='outer')
            all_chunks_data.append(chunk_df)
        
        current_start = current_end
        
    if not all_chunks_data:
        print("\nНе удалось собрать никаких данных для всего периода.")
        return pd.DataFrame()

    print("\nОбъединение данных из всех чанков...")
    final_df = pd.concat(all_chunks_data, axis=0)

    final_df = final_df[~final_df.index.duplicated(keep='first')]
    
    expected_column_names = list(queries_dict.keys())
    for col_name in expected_column_names:
        if col_name not in final_df.columns:
            final_df[col_name] = pd.NA 

    final_df = final_df.sort_index()

    return final_df

# --- Основной блок ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    CONFIG = load_config(CONFIG_FILE_PATH)

    artifacts_path_str = CONFIG.get('artifacts_dir', 'artifacts')
    artifacts_dir = BASE_DIR / artifacts_path_str
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Директория для артефактов: {artifacts_dir}")

    cache_dir = artifacts_dir / "prometheus_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        CACHE = Cache(str(cache_dir))
        print(f"Кеширование запросов Prometheus включено. Директория кеша: {cache_dir}")
    except Exception as e:
        print(f"Не удалось создать директорию кеша {cache_dir}: {e}. Кеширование отключено.")
        CACHE = None

    PROMETHEUS_URL = CONFIG.get('prometheus_url')
    QUERIES = CONFIG.get('queries')
    DATA_SETTINGS = CONFIG.get('data_settings', {})

    if not PROMETHEUS_URL or not QUERIES:
        print("Ошибка: 'prometheus_url' или 'queries' не найдены в файле конфигурации.")
        exit(1)

    STEP = DATA_SETTINGS.get('step', '30s')
    CHUNK_HOURS = DATA_SETTINGS.get('cache_chunk_hours', 1)
    PARQUET_FILENAME = DATA_SETTINGS.get('output_filename', 'prometheus_metrics_data.parquet')
    output_file_path = artifacts_dir / PARQUET_FILENAME

    print(f"\nСбор данных из Prometheus: {PROMETHEUS_URL}")
    print(f"Шаг: {STEP} | Размер чанка для кеширования: {CHUNK_HOURS} час(а)")
    print(f"Файл для сохранения: {output_file_path}")
    print("-" * 30)

    training_data_list = []
    collection_periods = DATA_SETTINGS.get('collection_periods_iso')

    if collection_periods and isinstance(collection_periods, list):
        print("Обнаружена конфигурация с несколькими периодами 'collection_periods_iso'.")
        for i, period in enumerate(collection_periods):
            try:
                start_time = datetime.fromisoformat(period['start'])
                end_time = datetime.fromisoformat(period['end'])
                if start_time >= end_time:
                    print(f"Ошибка в периоде {i+1}: 'start' ({start_time}) должен быть раньше 'end' ({end_time}). Пропуск периода.")
                    continue
                
                print(f"\n--- Обработка периода {i+1}/{len(collection_periods)} ---")
                period_df = collect_training_data(PROMETHEUS_URL, QUERIES, start_time, end_time, STEP, CHUNK_HOURS)
                if not period_df.empty:
                    training_data_list.append(period_df)

            except (KeyError, ValueError) as e:
                print(f"Ошибка в конфигурации периода {i+1}: {e}. Пропуск периода.")
                continue
    else:
        print("Конфигурация 'collection_periods_iso' не найдена, используется 'collection_period_hours' или 'start_time_iso'/'end_time_iso'.")
        collection_period_hours = DATA_SETTINGS.get('collection_period_hours')
        start_time_iso = DATA_SETTINGS.get('start_time_iso')
        end_time_iso = DATA_SETTINGS.get('end_time_iso')
        
        start_time, end_time = None, None
        
        if start_time_iso and end_time_iso and (not collection_period_hours or collection_period_hours == 0):
            try:
                start_time = datetime.fromisoformat(start_time_iso)
                end_time = datetime.fromisoformat(end_time_iso)
                if start_time >= end_time:
                    print("Ошибка: 'start_time_iso' должен быть раньше 'end_time_iso'.")
                    exit(1)
            except ValueError:
                print("Ошибка: 'start_time_iso' или 'end_time_iso' имеют неверный формат. Используйте YYYY-MM-DDTHH:MM:SS.")
                exit(1)
        elif collection_period_hours and collection_period_hours > 0:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=collection_period_hours)
        else:
            print("Ошибка: Некорректные настройки времени. Укажите 'collection_periods_iso', 'collection_period_hours' > 0 или 'start_time_iso' и 'end_time_iso'.")
            exit(1)
            
        if start_time and end_time:
            print(f"\n--- Обработка единого периода ---")
            single_period_df = collect_training_data(PROMETHEUS_URL, QUERIES, start_time, end_time, STEP, CHUNK_HOURS)
            if not single_period_df.empty:
                training_data_list.append(single_period_df)

    if training_data_list:
        final_training_data = pd.concat(training_data_list)
        final_training_data = final_training_data.sort_index()
        final_training_data = final_training_data[~final_training_data.index.duplicated(keep='first')]

        final_training_data['day_of_week'] = final_training_data.index.dayofweek.astype(int)
        final_training_data['hour_of_day'] = final_training_data.index.hour.astype(int)

        print("\n--- Собранные данные (первые 5 и последние 5 строк) ---")
        print(final_training_data.head())
        print("...")
        print(final_training_data.tail())
        print(f"\nРазмер итогового DataFrame: {final_training_data.shape}")

        try:
            final_training_data.to_parquet(output_file_path, engine='pyarrow', index=True)
            print(f"\nДанные успешно сохранены в {output_file_path}")
        except Exception as e:
            print(f"\nОшибка при сохранении данных в Parquet: {e}")
            print("Убедитесь, что библиотека 'pyarrow' (или 'fastparquet') установлена.")
    else:
        print("\nНе удалось собрать никаких данных для указанных периодов.")

    if CACHE is not None:
        CACHE.close()