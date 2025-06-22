# Deploying actuarial models in production

## Files

- `dataset.py`: Save the datasets in the data folder
- `main.py`: Fits the models and saves it them the models folder
- `app.py`: API to serve the models
- `requirements.txt`: Python packages required to run the project
- `Dockerfile`: Dockerfile to build the image
- `quote-page.py`: Streamlit app to get a quote from the model

## Set up the python environment

- Create a virtual environment, e.g. using venv: `python -m venv deployer`
- Activate the virtual environment: `source deployer/bin/activate`
- Install the required packages: `pip install -r requirements.txt`

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

then in the notebook folder you can execute the try_api.ipynb notebook to test the API.:

### Running the Docker container

To build the Docker image, run the following command:

```bash
docker build -t deployer .
```

To run the Docker container, use the following command:

```bash
docker run -d --rm --name deployer -p 8080:8080 deployer:latest
```

To stop the Docker container, use the following command:

```bash
docker stop deployer
```
