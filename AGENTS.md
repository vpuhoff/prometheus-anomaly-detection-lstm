# AGENTS.md

This repository contains scripts for collecting Prometheus metrics, preprocessing data,
training an LSTM autoencoder and detecting anomalies in real time.

## Style
- Python 3.12 is the target version.
- Follow PEP8: 4 spaces per indent and keep lines under 100 characters.
- Use type hints and include short docstrings for every function or class.
- Comments and log messages are generally in Russian—keep new ones consistent with this style.

## Setup
- Packages are managed via `uv`. If you don't have it installed, you can run:
  ```bash
  curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
  ```
- Before installing, create a virtual environment:
  ```bash
  uv venv
  ```
- Install dependencies from the lock file:
  ```bash
  uv pip sync requirements.lock.txt
  ```

## Dependencies
- After adding dependencies run `uv pip compile pyproject.toml --extra dev -o requirements.lock.txt`.

## Configuration
- All runtime settings are in `config.yaml`. Document any new configuration keys
both in this file and in `README.md`.

## Programmatic checks
Before committing changes:
1. Install dependencies:
   ```bash
   uv pip sync requirements.lock.txt
   ```
```