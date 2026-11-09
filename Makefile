# Thin wrapper around tasks.py (which also works without make on Windows).
# Phony targets always run their step; the file rules below only build what is missing,
# so `make train`, `make evaluate` or `make test` work on a fresh clone in any order.

.PHONY: all setup clean clean-data eda split train evaluate predict facts notebooks app test coverage lint report

PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))

CLEAN_DATA := data/interim/cars_clean.csv
TRAIN_DATA := data/processed/train.csv
PIPELINE   := models/final_pipeline.joblib

all:
	$(PYTHON) tasks.py all

setup:
	$(PYTHON) tasks.py setup

clean:
	$(PYTHON) tasks.py clean

clean-data:
	$(PYTHON) tasks.py clean-data

eda: $(CLEAN_DATA)
	$(PYTHON) tasks.py eda

split: $(CLEAN_DATA)
	$(PYTHON) tasks.py split

train: $(TRAIN_DATA)
	$(PYTHON) tasks.py train

evaluate: $(PIPELINE)
	$(PYTHON) tasks.py evaluate

predict: $(PIPELINE)
	$(PYTHON) tasks.py predict

facts:
	$(PYTHON) tasks.py facts

notebooks: $(PIPELINE)
	$(PYTHON) tasks.py notebooks

app: $(PIPELINE)
	$(PYTHON) tasks.py app

test: $(PIPELINE)
	$(PYTHON) tasks.py test

coverage: $(PIPELINE)
	$(PYTHON) tasks.py coverage

lint:
	$(PYTHON) tasks.py lint

report:
	$(PYTHON) tasks.py report

$(CLEAN_DATA):
	$(PYTHON) tasks.py clean-data

$(TRAIN_DATA): $(CLEAN_DATA)
	$(PYTHON) tasks.py split

$(PIPELINE): $(TRAIN_DATA)
	$(PYTHON) tasks.py train
