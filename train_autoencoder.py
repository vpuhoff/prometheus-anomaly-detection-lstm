import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Функции --- (load_config, build_lstm_autoencoder - остаются прежними)

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

def load_processed_data_parquet(file_path: Path) -> pd.DataFrame:
    """Загружает предобработанные данные из Parquet файла."""
    if not file_path.exists():
        logging.error(f"Ошибка: Файл данных Parquet не найден по пути {file_path}")
        exit(1)
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Данные из Parquet успешно загружены: {file_path}. Размер: {df.shape}")
        # df.info(verbose=True, show_counts=True) # Для более детальной информации можно раскомментировать
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных Parquet из {file_path}: {e}")
        exit(1)

def load_sequences_npy(file_path: Path) -> np.ndarray:
    """Загружает последовательности из NPY файла."""
    if not file_path.exists():
        logging.error(f"Ошибка: Файл NPY не найден по пути {file_path}")
        exit(1)
    try:
        sequences = np.load(file_path)
        logging.info(f"Последовательности из NPY успешно загружены: {file_path}. Форма: {sequences.shape}")
        return sequences
    except Exception as e:
        logging.error(f"Ошибка при загрузке NPY файла из {file_path}: {e}")
        exit(1)


def create_sequences_from_df(data_values: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Создает последовательности (окна) из массива значений временного ряда.
    Для автоэнкодера X и y - это одна и та же последовательность.
    """
    xs = []
    if len(data_values) < sequence_length:
        logging.warning(f"Данных ({len(data_values)}) меньше, чем длина последовательности ({sequence_length}). Невозможно создать последовательности.")
        return np.array(xs) # Возвращаем пустой массив, если данных не хватает

    for i in range(len(data_values) - sequence_length + 1):
        x = data_values[i:(i + sequence_length)]
        xs.append(x)
    return np.array(xs)

def build_lstm_autoencoder(sequence_length: int, num_features: int, config_params: dict) -> Model:
    """Строит LSTM автоэнкодер модель."""
    lstm_units_e1 = config_params.get('lstm_units_encoder1', 64)
    lstm_units_e2_latent = config_params.get('lstm_units_encoder2_latent', 32)
    lstm_units_d1 = config_params.get('lstm_units_decoder1', 32)
    lstm_units_d2 = config_params.get('lstm_units_decoder2', 64)

    inputs = Input(shape=(sequence_length, num_features))
    encoder = LSTM(lstm_units_e1, activation='relu', return_sequences=True)(inputs)
    encoder = LSTM(lstm_units_e2_latent, activation='relu', return_sequences=False)(encoder)
    bridge = RepeatVector(sequence_length)(encoder)
    decoder = LSTM(lstm_units_d1, activation='relu', return_sequences=True)(bridge)
    decoder = LSTM(lstm_units_d2, activation='relu', return_sequences=True)(decoder)
    outputs = TimeDistributed(Dense(num_features, activation='sigmoid'))(decoder)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_model(model_path: Path) -> tf.keras.Model:
    """Загружает модель TensorFlow/Keras из файла."""
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Модель успешно загружена из {model_path}")
        model.summary(print_fn=logging.info)
        return model
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели из {model_path}: {e}", exc_info=True)
        raise

# --- Основной блок ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    
    CONFIG = load_config(CONFIG_FILE_PATH)

    # Получение пути для артефактов ---
    artifacts_path_str = CONFIG.get('artifacts_dir', 'artifacts')
    artifacts_dir = BASE_DIR / artifacts_path_str
    # Создаем директорию, если она не существует
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Директория для артефактов: {artifacts_dir}")

    # Извлечение настроек
    preprocess_settings = CONFIG.get('preprocessing_settings', {})
    training_settings = CONFIG.get('training_settings', {})

    # Общие параметры обучения
    sequence_length = training_settings.get('sequence_length', 20)
    train_split_ratio = training_settings.get('train_split_ratio', 0.8)
    epochs = training_settings.get('epochs', 50)
    batch_size = training_settings.get('batch_size', 64)
    learning_rate = training_settings.get('learning_rate', 0.001)
    early_stopping_patience = training_settings.get('early_stopping_patience', 0)

    # Определение входных данных и имени выходной модели
    logging.info("Режим обучения: на предобработанных данных")
    input_parquet_filename = training_settings.get('input_processed_filename',
                                                  preprocess_settings.get('processed_output_filename'))
    if not input_parquet_filename:
        logging.error("Имя файла с предобработанными данными Parquet не указано в config.yaml "
                      "(training_settings.input_processed_filename или preprocessing_settings.processed_output_filename).")
        exit(1)
        
    # Формируем путь к входному файлу внутри директории артефактов ---
    input_file_path = artifacts_dir / input_parquet_filename
    df_processed = load_processed_data_parquet(input_file_path)

    if df_processed.empty:
        logging.error("Загруженный DataFrame для обучения пуст. Обучение невозможно.")
        exit(1)

    num_features = df_processed.shape[1]
    data_values = df_processed.values
    X_all_sequences = create_sequences_from_df(data_values, sequence_length)
    model_output_filename = training_settings.get('model_output_filename', 'lstm_autoencoder_model.keras')

    if X_all_sequences.shape[0] == 0:
        logging.error("Не удалось создать ни одной последовательности из Parquet данных. Обучение невозможно.")
        exit(1)

    # Формируем путь к выходному файлу модели внутри директории артефактов ---
    model_output_path = artifacts_dir / model_output_filename
    logging.info(f"Модель будет сохранена в: {model_output_path}")

    # Для автоэнкодера X (вход) и Y (цель) - это одни и те же последовательности
    y_all_sequences = X_all_sequences 
    logging.info(f"Всего создано/загружено {X_all_sequences.shape[0]} последовательностей длиной {X_all_sequences.shape[1]} с {X_all_sequences.shape[2]} признаками.")

    # Разделение данных на обучающую и валидационную выборки
    if X_all_sequences.shape[0] < 2 : # Нужно хотя бы 2 сэмпла для разделения
        logging.error(f"Слишком мало последовательностей ({X_all_sequences.shape[0]}) для разделения на train/validation.")
        exit(1)

    val_split_size = 1.0 - train_split_ratio
    if val_split_size <= 0.0 or val_split_size >= 1.0:
        if X_all_sequences.shape[0] > 1: # Если есть что валидировать
            logging.warning(f"train_split_ratio ({train_split_ratio}) некорректен для создания валидационной выборки. Используется одна последовательность для валидации, если возможно.")
            # Если train_split_ratio = 1, то val_split_size = 0. sklearn требует test_size > 0
            # В этом случае, можно взять 1 сэмпл для валидации, если всего их > 1
            if X_all_sequences.shape[0] > 1 and train_split_ratio >= 1.0 :
                 X_train, X_val, y_train, y_val = X_all_sequences[:-1], X_all_sequences[-1:], y_all_sequences[:-1], y_all_sequences[-1:]
            elif X_all_sequences.shape[0] > 1 and train_split_ratio <= 0.0 : # Все на валидацию
                 X_train, X_val, y_train, y_val = X_all_sequences[:1], X_all_sequences[1:], y_all_sequences[:1], y_all_sequences[1:]
            else: # Если всего один сэмпл, то он пойдет в трейн, валидации не будет
                 X_train, X_val, y_train, y_val = X_all_sequences, np.array([]).reshape(0,sequence_length,num_features), y_all_sequences, np.array([]).reshape(0,sequence_length,num_features)
                 logging.warning("Валидационная выборка пуста, так как недостаточно данных.")
        else:
            X_train, X_val, y_train, y_val = X_all_sequences, np.array([]).reshape(0,sequence_length,num_features), y_all_sequences, np.array([]).reshape(0,sequence_length,num_features)
            logging.warning("Валидационная выборка пуста, так как всего одна последовательность.")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all_sequences, y_all_sequences, train_size=train_split_ratio, shuffle=True, random_state=42
        )

    logging.info(f"Размер обучающей выборки: {X_train.shape}")
    logging.info(f"Размер валидационной выборки: {X_val.shape}")

    # Построение модели LSTM автоэнкодера
    autoencoder_model = build_lstm_autoencoder(sequence_length, num_features, training_settings)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder_model.compile(optimizer=optimizer, loss='mse')
    autoencoder_model.summary(print_fn=logging.info)

    # Обучение модели
    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1)
        callbacks.append(early_stopping)
        logging.info(f"EarlyStopping включен с терпением: {early_stopping_patience} эпох.")
    
    # Директория для чекпоинтов внутри директории артефактов ---
    checkpoint_dir = artifacts_dir / "model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = checkpoint_dir / "best_model.keras"
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1)
    callbacks.append(model_checkpoint_callback)
    logging.info(f"ModelCheckpoint включен. Лучшая модель будет сохранена в: {checkpoint_filepath}")

    logging.info("\nНачало обучения модели...")
    validation_data_to_pass = None
    if X_val.shape[0] > 0:
        validation_data_to_pass = (X_val, y_val)

    history = autoencoder_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data_to_pass,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    logging.info("Обучение завершено.")

    if checkpoint_filepath.exists() and any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        logging.info(f"Загрузка лучшей модели из чекпоинта: {checkpoint_filepath}")
        try:
            autoencoder_model = load_model(checkpoint_filepath)
        except Exception as e:
            logging.error(f"Ошибка загрузки модели из чекпоинта {checkpoint_filepath}: {e}. Сохраняется последняя модель.")

    try:
        autoencoder_model.save(model_output_path)
        logging.info(f"Обученная модель сохранена в: {model_output_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели: {e}")

    # Визуализация истории обучения (потери)
    if validation_data_to_pass: 
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Потери на обучении (Train Loss)')
        plt.plot(history.history['val_loss'], label='Потери на валидации (Validation Loss)')
        plt.title(f'История обучения ({model_output_filename}): функция потерь')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери (MSE)')
        plt.legend()
        plt.grid(True)
        # Путь для сохранения графика внутри директории артефактов ---
        plot_filename_loss = artifacts_dir / f"training_history_loss_{model_output_filename.replace('.keras', '')}.png"
        try:
            plt.savefig(plot_filename_loss)
            logging.info(f"График истории обучения сохранен в: {plot_filename_loss}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении графика истории обучения: {e}")
    else:
        logging.info("График val_loss не строится, так как валидационная выборка была пуста.")

    # Оценка распределения ошибок реконструкции на валидационных данных (если они есть)
    if X_val.shape[0] > 0:
        logging.info("\nОценка ошибок реконструкции на валидационной выборке...")
        X_val_pred = autoencoder_model.predict(X_val, batch_size=batch_size)
        mse_val = np.mean(np.power(X_val - X_val_pred, 2), axis=(1, 2))

        plt.figure(figsize=(10, 6))
        plt.hist(mse_val, bins=50, density=True, alpha=0.75)
        plt.title(f'Гистограмма ошибок реконструкции ({model_output_filename}) на валидации (MSE)')
        plt.xlabel('Ошибка реконструкции (MSE)')
        plt.ylabel('Плотность')
        plt.grid(True)
        # Путь для сохранения гистограммы внутри директории артефактов ---
        plot_filename_hist = artifacts_dir / f"reconstruction_error_histogram_{model_output_filename.replace('.keras', '')}.png"
        try:
            plt.savefig(plot_filename_hist)
            logging.info(f"Гистограмма ошибок реконструкции сохранена в: {plot_filename_hist}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении гистограммы ошибок: {e}")
    else:
        logging.info("Гистограмма ошибок реконструкции на валидации не строится, так как валидационная выборка была пуста.")

    logging.info(f"--- Скрипт обучения модели '{model_output_filename}' завершен ---")