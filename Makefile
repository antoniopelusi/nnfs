.SILENT: all clean setup run_blobs_classification run_blobs_classification_dp run_iris_classification run_iris_classification_dp run_moons_classification run_moons_classification_dp run_regression_sine run_regression_sine_dp

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

all:
	echo "|> No target selected. Abort."

# ----------------------------
# Clean venv and modules
# ----------------------------
clean:
	echo "=========| clean started... |========="
	rm -rf .venv
	echo "=========| clean completed |========="

# ----------------------------
# Reset and setup environment and project
# ----------------------------
setup: clean
	echo "=========| setup started... |========="
	python3 -m venv $(VENV);
	echo "|> installing required libraries..."
	$(PIP) install -r requirements.txt
	echo "=========| setup completed |========="

# ----------------------------
# Run
# ----------------------------
run_blobs_classification:
	echo "=========| run started... |========="
	$(PYTHON) test_blobs_classification.py;
	echo "=========| run completed |========="

run_blobs_classification_dp:
	echo "=========| run started... |========="
	$(PYTHON) test_blobs_classification_dp.py;
	echo "=========| run completed |========="

run_iris_classification:
	echo "=========| run started... |========="
	$(PYTHON) test_iris_classification.py;
	echo "=========| run completed |========="

run_iris_classification_dp:
	echo "=========| run started... |========="
	$(PYTHON) test_iris_classification_dp.py;
	echo "=========| run completed |========="

run_moons_classification:
	echo "=========| run started... |========="
	$(PYTHON) test_moons_classification.py;
	echo "=========| run completed |========="

run_moons_classification_dp:
	echo "=========| run started... |========="
	$(PYTHON) test_moons_classification_dp.py;
	echo "=========| run completed |========="

run_regression_sine:
	echo "=========| run started... |========="
	$(PYTHON) test_regression_sine.py;
	echo "=========| run completed |========="

run_regression_sine_dp:
	echo "=========| run started... |========="
	$(PYTHON) test_regression_sine_dp.py;
	echo "=========| run completed |========="
