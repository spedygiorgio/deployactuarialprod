# Deploying actuarial models in production

## Repository structure

- `app.py`: FastAPI application (endpoints, lifecycle, logging)
- `schemas.py`: Pydantic request/response models for the API
- `constants.py`: Centralized validation constants for categorical and numeric variables
- `dataset.py`: generates and stores train/validation/test datasets in `data/`
- `main.py`: runs training pipeline and persists CatBoost models in `models/`
- `steps/`: training and prediction pipeline components
- `utils.py`: shared utilities (logging and timer helpers)
- `quote-page.py`: Streamlit demo app for quote simulation
- `notebooks/try_api.ipynb`: notebook to test API endpoints
- `requirements.txt`: Python dependencies
- `Dockerfile`: container image definition for API deployment
- `models/`: trained CatBoost models used at inference time
- `data/`: input and split datasets
- `docs/`: presentation materials

## Set up the python environment

### approccio tradizionale

- Create a virtual environment, e.g. using venv: `python -m venv deployer`

- Activate the virtual environment (Windows): `deployer\Scripts\activate.bat`
- Install the required packages: `pip install -r requirements.txt`

### oppure con `uv`

```bash
uv venv deployer
deployer\Scripts\activate.bat
uv pip install -r requirements.txt
```

## How to run

- Clone the repository

### Fitting the models

From another terminal, run the following commands:

```bash
mlflow ui
```

This will start the MLflow UI at `http://localhost:5000`.

Then, in the first terminal, run the following commands:

- Execute `python dataset.py` to save the datasets
- Execute `python main.py` to fit the models

### Trying the models from the streamlit app

From another terminal, run the following command:

```bash
streamlit run quote-page.py
```

### Running the API

From another terminal, run the following command:

```bash
python app.py
```

Then execute `notebooks/try_api.ipynb` to test the endpoints (`/health`, `/ready`, `/predict/`).

### Running the Docker container

To build the Docker image, run the following command:

```bash
docker build -t deployer:latest .
```

To run the Docker container, use the following command:

```bash
docker run -d --rm --name deployer -p 8080:8080 deployer:latest
```

To stop the Docker container, use the following command:

```bash
docker stop deployer
```

## API endpoints

- `GET /health`: liveness check (process is running)
- `GET /ready`: readiness check (models loaded and service ready)
- `POST /predict/`: returns `Frequency`, `Severity`, and `Pure_Premium`
