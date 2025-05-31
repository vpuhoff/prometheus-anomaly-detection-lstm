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
    return {} # Возвращаем пустой словарь в случае другой ошибки

def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Загружает предобработанные данные из Parquet файла."""
    if not file_path.exists():
        print(f"Ошибка: Файл данных не найден по пути {file_path}")
        exit(1)
    try:
        df = pd.read_parquet(file_path)
        print(f"Предобработанные данные успешно загружены из {file_path}. Размер: {df.shape}")
        df.info()
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных из {file_path}: {e}")
        exit(1)

def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Создает последовательности (окна) из временного ряда.
    Для автоэнкодера X и y - это одна и та же последовательность.
    """
    xs = []
    for i in range(len(data) - sequence_length + 1): # +1 чтобы включить последний возможный старт
        x = data[i:(i + sequence_length)]
        xs.append(x)
    return np.array(xs)

def build_lstm_autoencoder(sequence_length: int, num_features: int, config_params: dict) -> Model:
    """Строит LSTM автоэнкодер модель."""
    
    # Параметры архитектуры из конфига
    lstm_units_e1 = config_params.get('lstm_units_encoder1', 64)
    lstm_units_e2_latent = config_params.get('lstm_units_encoder2_latent', 32)
    lstm_units_d1 = config_params.get('lstm_units_decoder1', 32)
    lstm_units_d2 = config_params.get('lstm_units_decoder2', 64)

    # Кодировщик
    inputs = Input(shape=(sequence_length, num_features))
    # tf.keras.layers.Masking(mask_value=0., input_shape=(sequence_length, num_features))) # Если есть паддинг
    encoder = LSTM(lstm_units_e1, activation='relu', return_sequences=True)(inputs)
    encoder = LSTM(lstm_units_e2_latent, activation='relu', return_sequences=False)(encoder) # Латентное представление

    # Мост (повторяем вектор латентного представления для каждого временного шага)
    bridge = RepeatVector(sequence_length)(encoder)

    # Декодировщик
    decoder = LSTM(lstm_units_d1, activation='relu', return_sequences=True)(bridge)
    decoder = LSTM(lstm_units_d2, activation='relu', return_sequences=True)(decoder)
    outputs = TimeDistributed(Dense(num_features, activation='sigmoid'))(decoder) # 'sigmoid' т.к. данные отмасштабированы MinMaxScaler [0,1]

    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Основной блок ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    
    CONFIG = load_config(CONFIG_FILE_PATH)

    preprocess_settings = CONFIG.get('preprocessing_settings', {})
    training_settings = CONFIG.get('training_settings', {})

    # Входной файл (результат работы скрипта предобработки)
    input_filename_processed = training_settings.get('input_processed_filename', 
                                               preprocess_settings.get('processed_output_filename'))
    if not input_filename_processed:
        print("Ошибка: Имя файла с предобработанными данными не указано в конфигурации.")
        exit(1)
    input_file_path = BASE_DIR / input_filename_processed

    # Параметры модели и обучения
    model_output_filename = training_settings.get('model_output_filename', 'lstm_autoencoder_model.keras')
    model_output_path = BASE_DIR / model_output_filename
    
    sequence_length = training_settings.get('sequence_length', 20)
    train_split_ratio = training_settings.get('train_split_ratio', 0.8)
    epochs = training_settings.get('epochs', 50)
    batch_size = training_settings.get('batch_size', 64)
    learning_rate = training_settings.get('learning_rate', 0.001)
    early_stopping_patience = training_settings.get('early_stopping_patience', 0) # 0 или null для отключения

    print("--- Начало скрипта обучения автоэнкодера ---")

    # 1. Загрузка предобработанных данных
    df_processed = load_processed_data(input_file_path)
    if df_processed.empty:
        print("Загруженный DataFrame пуст. Обучение невозможно.")
        exit(1)
    
    num_features = df_processed.shape[1]
    data_values = df_processed.values # Получаем NumPy массив

    # 2. Создание последовательностей
    # Для автоэнкодера X (вход) и Y (цель) - это одни и те же последовательности
    sequences = create_sequences(data_values, sequence_length)
    if len(sequences) == 0:
        print(f"Не удалось создать последовательности. Возможно, данных ({len(data_values)} точек) "
              f"меньше, чем длина последовательности ({sequence_length}).")
        exit(1)
        
    X = sequences
    y = sequences # Автоэнкодер учится восстанавливать вход
    print(f"Создано {X.shape[0]} последовательностей длиной {X.shape[1]} с {X.shape[2]} признаками.")

    # 3. Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_split_ratio, shuffle=True, random_state=42)
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")

    # 4. Построение модели LSTM автоэнкодера
    autoencoder_model = build_lstm_autoencoder(sequence_length, num_features, training_settings)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder_model.compile(optimizer=optimizer, loss='mse') # Mean Squared Error для реконструкции
    autoencoder_model.summary()

    # 5. Обучение модели
    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        callbacks.append(early_stopping)
        print(f"EarlyStopping включен с терпением: {early_stopping_patience} эпох.")
    
    # ModelCheckpoint для сохранения лучшей модели (опционально)
    # checkpoint_filepath = BASE_DIR / 'best_model_checkpoint.keras'
    # model_checkpoint_callback = ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=False, # Сохраняем всю модель
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True) # Сохраняем только лучшую модель
    # callbacks.append(model_checkpoint_callback)


    print("\nНачало обучения модели...")
    history = autoencoder_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=True
    )

    print("Обучение завершено.")

    # 6. Сохранение обученной модели
    try:
        autoencoder_model.save(model_output_path)
        print(f"Обученная модель сохранена в: {model_output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")

    # 7. Визуализация истории обучения (потери)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Потери на обучении (Train Loss)')
    plt.plot(history.history['val_loss'], label='Потери на валидации (Validation Loss)')
    plt.title('История обучения: функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери (MSE)')
    plt.legend()
    plt.grid(True)
    plot_filename = BASE_DIR / "training_history_loss.png"
    try:
        plt.savefig(plot_filename)
        print(f"График истории обучения сохранен в: {plot_filename}")
    except Exception as e:
        print(f"Ошибка при сохранении графика: {e}")
    # plt.show() # Раскомментируйте, если хотите показать график сразу

    # (Опционально) Оценка распределения ошибок реконструкции на валидационных данных
    print("\nОценка ошибок реконструкции на валидационной выборке...")
    X_val_pred = autoencoder_model.predict(X_val)
    mse_val = np.mean(np.power(X_val - X_val_pred, 2), axis=(1, 2)) # Средняя ошибка для каждой последовательности

    plt.figure(figsize=(10, 6))
    plt.hist(mse_val, bins=50, density=True, alpha=0.75)
    plt.title('Гистограмма ошибок реконструкции на валидационной выборке (MSE)')
    plt.xlabel('Ошибка реконструкции (MSE)')
    plt.ylabel('Плотность')
    plt.grid(True)
    reconstruction_error_plot_filename = BASE_DIR / "reconstruction_error_validation_histogram.png"
    try:
        plt.savefig(reconstruction_error_plot_filename)
        print(f"Гистограмма ошибок реконструкции сохранена в: {reconstruction_error_plot_filename}")
    except Exception as e:
        print(f"Ошибка при сохранении гистограммы ошибок: {e}")
    # plt.show()

    print("\n--- Скрипт обучения автоэнкодера завершен ---")