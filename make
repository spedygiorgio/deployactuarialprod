"python" = "deployer/Scripts/python"
"pip" = "deployer/Scripts/pip"

setup:
    $(python) -m venv deployer
    $(pip) install --upgrade pip
    $(pip) install -r requirements.txt

run:
    $(python) main.py

mlflow:
    $(python) -m mlflow ui

test:
    $(python) -m pytest

clean:
    rm -rf steps/__pycache__
    rm -rf tests/__pycache__

remove:
    rm -rf deployer