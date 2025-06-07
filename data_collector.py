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

    return final_df

# --- Основной блок ---
if __name__ == "__main__":
    # Определяем путь к файлу конфигурации относительно текущего скрипта
    CONFIG_FILE_PATH = Path(__file__).parent / "config.yaml"
    CONFIG = load_config(CONFIG_FILE_PATH)

    PROMETHEUS_URL = CONFIG.get('prometheus_url')
    QUERIES = CONFIG.get('queries')
    DATA_SETTINGS = CONFIG.get('data_settings', {})

    if not PROMETHEUS_URL or not QUERIES:
        print("Ошибка: 'prometheus_url' или 'queries' не найдены в файле конфигурации.")
        exit(1)

    STEP = DATA_SETTINGS.get('step', '30s')
    PARQUET_FILENAME = DATA_SETTINGS.get('output_filename', 'prometheus_metrics_data.parquet')
    
    print(f"Сбор данных из Prometheus: {PROMETHEUS_URL}")
    print(f"Шаг: {STEP}")
    print(f"Файл для сохранения: {PARQUET_FILENAME}")
    print("-" * 30)

    training_data_list = []
    collection_periods = DATA_SETTINGS.get('collection_periods_iso')

    # 1. Проверяем новый параметр collection_periods_iso
    if collection_periods and isinstance(collection_periods, list):
        print("Обнаружена конфигурация с несколькими периодами 'collection_periods_iso'.")
        for i, period in enumerate(collection_periods):
            try:
                start_time = datetime.fromisoformat(period['start'])
                end_time = datetime.fromisoformat(period['end'])
                if start_time >= end_time:
                    print(f"Ошибка в периоде {i+1}: 'start' ({start_time}) должен быть раньше 'end' ({end_time}). Пропуск периода.")
                    continue
                
                print(f"\n--- Сбор данных для периода {i+1}/{len(collection_periods)} ---")
                period_df = collect_training_data(PROMETHEUS_URL, QUERIES, start_time, end_time, STEP)
                if not period_df.empty:
                    training_data_list.append(period_df)

            except (KeyError, ValueError) as e:
                print(f"Ошибка в конфигурации периода {i+1}: {e}. Убедитесь, что 'start' и 'end' заданы в верном формате ISO. Пропуск периода.")
                continue

    # 2. Если новый параметр не найден, используем старую логику
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
            print(f"\n--- Сбор данных для периода с {start_time.strftime('%Y-%m-%d %H:%M:%S')} по {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            single_period_df = collect_training_data(PROMETHEUS_URL, QUERIES, start_time, end_time, STEP)
            if not single_period_df.empty:
                training_data_list.append(single_period_df)

    # 3. Собираем итоговый DataFrame из всех собранных частей
    if training_data_list:
        print("\nОбъединение данных из всех периодов...")
        final_training_data = pd.concat(training_data_list)
        
        # Сортируем по индексу (времени) на случай, если периоды шли не по порядку
        final_training_data = final_training_data.sort_index()
        
        # Удаляем возможные дубликаты по индексу, если периоды перекрывались
        final_training_data = final_training_data[~final_training_data.index.duplicated(keep='first')]

        # Добавляем временные признаки в итоговый DataFrame
        final_training_data['day_of_week'] = final_training_data.index.dayofweek.astype(int)
        final_training_data['hour_of_day'] = final_training_data.index.hour.astype(int)

        print("\n--- Собранные данные (первые 5 и последние 5 строк) ---")
        print(final_training_data.head())
        print("...")
        print(final_training_data.tail())
        print(f"\nРазмер итогового DataFrame: {final_training_data.shape}")
        print("\nИнформация о DataFrame (до сохранения):")
        final_training_data.info()

        try:
            output_file_path = Path(__file__).parent / PARQUET_FILENAME
            final_training_data.to_parquet(output_file_path, engine='pyarrow', index=True)
            print(f"\nДанные успешно сохранены в {output_file_path}")
        except Exception as e:
            print(f"\nОшибка при сохранении данных в Parquet: {e}")
            print("Убедитесь, что библиотека 'pyarrow' (или 'fastparquet') установлена.")
    else:
        print("\nНе удалось собрать никаких данных для указанных периодов.")