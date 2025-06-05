# AGENTS.md

This repository contains scripts for collecting Prometheus metrics, preprocessing data,
training an LSTM autoencoder and detecting anomalies in real time.

## Style
- Python 3.12 is the target version.
- Follow PEP8: 4 spaces per indent and keep lines under 100 characters.
- Use type hints and include short docstrings for every function or class.
- Comments and log messages are generally in Russian—keep new ones consistent with this style.

## Dependencies
- Packages are managed via `Pipfile` / `Pipfile.lock`.
- After adding dependencies run `pipenv lock` to regenerate `Pipfile.lock`.

## Configuration
- All runtime settings are in `config.yaml`. Document any new configuration keys
both in this file and in `README.md`.

## Programmatic checks
Before committing changes:
1. Install dependencies:
   ```bash
   pipenv install --dev
