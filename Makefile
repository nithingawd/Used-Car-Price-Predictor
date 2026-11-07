.PHONY: setup clean clean-data eda split train evaluate predict app test lint

PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))

setup:
	$(PYTHON) tasks.py setup

clean:
	$(PYTHON) tasks.py clean

clean-data:
	$(PYTHON) tasks.py clean-data

eda:
	$(PYTHON) tasks.py eda

split:
	$(PYTHON) tasks.py split

train:
	$(PYTHON) tasks.py train

evaluate:
	$(PYTHON) tasks.py evaluate

predict:
	$(PYTHON) tasks.py predict

app:
	$(PYTHON) tasks.py app

test:
	$(PYTHON) tasks.py test

lint:
	$(PYTHON) tasks.py lint

