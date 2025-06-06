import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> None:
    """Запускает указанный скрипт в отдельном процессе."""
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Не найден скрипт: {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(
        description="Утилита для запуска этапов обработки данных"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Сбор данных из Prometheus")
    subparsers.add_parser("preprocess", help="Предобработка собранных данных")
    subparsers.add_parser("train", help="Обучение модели")
    subparsers.add_parser("detect", help="Запуск realtime детектора")

    args = parser.parse_args()

    if args.command == "collect":
        run_script("data_collector.py")
    elif args.command == "preprocess":
        run_script("preprocess_data.py")
    elif args.command == "train":
        run_script("train_autoencoder.py")
    elif args.command == "detect":
        run_script("realtime_detector.py")


if __name__ == "__main__":
    main()
