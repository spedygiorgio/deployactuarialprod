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
- Execute `python dataset.py` to save the datasets
- Execute `python main.py` to fit the models