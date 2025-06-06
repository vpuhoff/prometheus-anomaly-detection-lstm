import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib # Для сохранения и загрузки скейлера
import numpy as np # Для np.nan при необходимости

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

def load_data(file_path: Path) -> pd.DataFrame:
    """Загружает данные из Parquet файла."""
    if not file_path.exists():
        print(f"Ошибка: Файл данных не найден по пути {file_path}")
        exit(1)
    try:
        df = pd.read_parquet(file_path)
        print(f"Данные успешно загружены из {file_path}. Размер: {df.shape}")
        df.info()
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных из {file_path}: {e}")
        exit(1)

def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Обрабатывает пропущенные значения в DataFrame."""
    print(f"\nОбработка пропущенных значений (NaN) по стратегии: {strategy}")
    print(f"Количество NaN до обработки:\n{df.isnull().sum()}")

    if strategy == "ffill_then_bfill":
        df_filled = df.ffill().bfill()
    elif strategy == "mean":
        # Убедимся, что применяем только к числовым колонкам
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                print(f"Предупреждение: Колонка {col} не является числовой, пропуск для 'mean' fill.")
        df_filled = df # df был изменен на месте
    elif strategy == "median":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                 print(f"Предупреждение: Колонка {col} не является числовой, пропуск для 'median' fill.")
        df_filled = df # df был изменен на месте
    elif strategy == "drop_rows":
        df_filled = df.dropna()
    elif strategy == "none":
        print("Пропуски NaN не обрабатывались согласно конфигурации.")
        df_filled = df
    else:
        print(f"Предупреждение: Неизвестная стратегия заполнения NaN '{strategy}'. Пропуски не обработаны.")
        df_filled = df
    
    print(f"Количество NaN после обработки:\n{df_filled.isnull().sum()}")
    if df_filled.isnull().sum().sum() > 0:
        print("ВНИМАНИЕ: После обработки все еще остались NaN значения. Проверьте данные и стратегию.")
    return df_filled

def scale_data(df: pd.DataFrame, scaler_type: str, scaler_output_path: Path) -> (pd.DataFrame, object):
    """Масштабирует данные и сохраняет скейлер."""
    print(f"\nМасштабирование данных с использованием: {scaler_type}")
    # Предполагаем, что все колонки в df - это метрики, которые нужно масштабировать
    # Если есть неметрические колонки (кроме индекса), их нужно отфильтровать
    metric_columns = df.columns
    data_to_scale = df[metric_columns].values # Получаем NumPy массив для скейлера

    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "StandardScaler":
        scaler = StandardScaler()
    else:
        print(f"Предупреждение: Неизвестный тип скейлера '{scaler_type}'. Используется MinMaxScaler по умолчанию.")
        scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data_to_scale)
    df_scaled = pd.DataFrame(scaled_data, columns=metric_columns, index=df.index)

    try:
        joblib.dump(scaler, scaler_output_path)
        print(f"Скейлер сохранен в: {scaler_output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении скейлера: {e}")

    return df_scaled, scaler

def save_processed_data(df: pd.DataFrame, file_path: Path):
    """Сохраняет обработанный DataFrame в Parquet файл."""
    try:
        df.to_parquet(file_path, index=True)
        print(f"\nОбработанные данные успешно сохранены в: {file_path}")
        df.info()
    except Exception as e:
        print(f"Ошибка при сохранении обработанных данных в {file_path}: {e}")

# --- Основной блок ---
if __name__ == "__main__":
    # Определяем путь к файлу конфигурации
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    
    CONFIG = load_config(CONFIG_FILE_PATH)

    # Получаем настройки из конфигурации
    data_settings = CONFIG.get('data_settings', {})
    preprocess_settings = CONFIG.get('preprocessing_settings', {})

    # Входной файл (результат работы предыдущего скрипта)
    input_filename = preprocess_settings.get('input_filename', data_settings.get('output_filename'))
    if not input_filename:
        print("Ошибка: Имя входного файла не указано ни в 'preprocessing_settings.input_filename', ни в 'data_settings.output_filename'.")
        exit(1)
    input_file_path = BASE_DIR / input_filename

    # Настройки предобработки
    nan_strategy = preprocess_settings.get('nan_fill_strategy', 'ffill_then_bfill')
    scaler_type_config = preprocess_settings.get('scaler_type', 'MinMaxScaler')
    
    # Выходные файлы
    processed_output_filename = preprocess_settings.get('processed_output_filename', 'processed_metrics_data.parquet')
    processed_output_file_path = BASE_DIR / processed_output_filename
    
    scaler_output_filename = preprocess_settings.get('scaler_output_filename', 'fitted_scaler.joblib')
    scaler_output_file_path = BASE_DIR / scaler_output_filename

    print("--- Начало скрипта предобработки данных ---")

    # 1. Загрузка данных
    raw_df = load_data(input_file_path)
    if raw_df.empty:
        print("Загруженный DataFrame пуст. Предобработка невозможна.")
        exit(1)

    # 2. Обработка пропущенных значений
    # Копируем DataFrame, чтобы избежать SettingWithCopyWarning при модификации
    df_processed = raw_df.copy()
    df_processed = handle_missing_values(df_processed, nan_strategy)

    # Добавляем признаки дня недели и часа суток
    df_processed['day_of_week'] = df_processed.index.dayofweek.astype(int)
    df_processed['hour_of_day'] = df_processed.index.hour.astype(int)
    
    if df_processed.empty and not raw_df.empty:
        print("DataFrame стал пустым после обработки NaN (например, из-за drop_rows). Завершение.")
        exit(1)
    if df_processed.isnull().values.any():
         print("ПРЕДУПРЕЖДЕНИЕ: В данных все еще есть NaN после этапа обработки пропусков. Это может вызвать проблемы при масштабировании или обучении.")


    # 3. Масштабирование данных
    df_scaled, fitted_scaler = scale_data(df_processed, scaler_type_config, scaler_output_file_path)
    print("\nПервые 5 строк отмасштабированных данных:")
    print(df_scaled.head())

    # 4. Сохранение обработанных данных
    save_processed_data(df_scaled, processed_output_file_path)
    
    print("\n--- Скрипт предобработки данных завершен ---")