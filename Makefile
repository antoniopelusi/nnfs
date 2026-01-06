.SILENT: all clean setup run

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
run:
	echo "=========| run started... |========="
	$(PYTHON) nn.py;
	echo "=========| run completed |========="
