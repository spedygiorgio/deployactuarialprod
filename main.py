#%% libraries
import logging
import yaml
import mlflow
import mlflow.catboost
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from catboost import CatBoostRegressor
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

#%% paths
os.makedirs('models', exist_ok=True)
freq_model_path = os.path.join('models', 'frequency_model.cbm')
sev_model_path = os.path.join('models', 'severity_model.cbm')

#%% main core
def main():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    # loading paths
    freq_model_path = config['models']['frequency']
    sev_model_path = config['models']['severity']

    with mlflow.start_run() as run:
        # load the datasets
        ingestor = Ingestion()
        ## frequency
        freq_train, freq_valid, freq_test = ingestor.load_freq()
        logging.info(f'Frequency datasets loaded: train={freq_train.shape}, valid={freq_valid.shape}, test={freq_test.shape}')
        ## severity datasets
        severity_train, severity_valid, severity_test = ingestor.load_severity()
        logging.info(f'Severity datasets loaded: train={severity_train.shape}, valid={severity_valid.shape}, test={severity_test.shape}')
        # clean the datasets
        cleaner = Cleaner()
        ## frequency
        freq_train = cleaner.clean(freq_train, 'frequency')
        freq_valid = cleaner.clean(freq_valid, 'frequency')
        freq_test = cleaner.clean(freq_test, 'frequency')
        logging.info(f'Frequency datasets cleaned: train={freq_train.shape}, valid={freq_valid.shape}, test={freq_test.shape}')
        ## severity
        severity_train = cleaner.clean(severity_train, 'severity')
        severity_valid = cleaner.clean(severity_valid, 'severity')
        severity_test = cleaner.clean(severity_test, 'severity')
        logging.info(f'Severity datasets cleaned: train={severity_train.shape}, valid={severity_valid.shape}, test={severity_test.shape}')
        
        # train the models
        ## frequency
        trainer_frequency = Trainer(target='ClaimNb', train_data=freq_train, val_data=freq_valid, test_data=freq_test)
        _, _, test_freq_pool = trainer_frequency.create_pools()
        frequency_model = trainer_frequency.train_model()
        freq_apratio, freq_mpd = trainer_frequency.evaluate_model(frequency_model, test_freq_pool)
        mlflow.log_metric('freq_apratio', freq_apratio)
        mlflow.log_metric('freq_mpd', freq_mpd)
        logging.info(f'Frequency model trained: AP ratio={freq_apratio}, MPD={freq_mpd}')
        frequency_model.save_model(freq_model_path)
        
        ## severity
        trainer_severity = Trainer(target='severity', train_data=severity_train, val_data=severity_valid, test_data=severity_test, model_type='severity')
        _, _, test_sev_pool = trainer_severity.create_pools()
        severity_model = trainer_severity.train_model()
        sev_apratio, sev_rmse = trainer_severity.evaluate_model(severity_model, test_sev_pool)
        mlflow.log_metric('sev_apratio', sev_apratio)
        mlflow.log_metric('sev_rmse', sev_rmse)
        logging.info(f'Severity model trained: AP ratio={sev_apratio}, RMSE={sev_rmse}')
        severity_model.save_model(sev_model_path)

        # tagging models on MLflow
        ## tagging
        mlflow.set_tag('modelli', 'catboost freqsev')
        ## logging
        mlflow.catboost.log_model(frequency_model, 'frequency_model')
        mlflow.catboost.log_model(severity_model, 'severity_model')
        ## registering
        ### frequency
        frequency_model_name = 'catboost frequency model'
        model_uri = f'runs:/{run.info.run_id}/frequency_model'
        mlflow.register_model(model_uri, frequency_model_name)
        ### severity
        severity_model_name = 'catboost severity model'
        model_uri = f'runs:/{run.info.run_id}/severity_model'
        mlflow.register_model(model_uri, severity_model_name)

        # predict the test data
        ## frequency
        reload_freq_model = CatBoostRegressor().load_model(freq_model_path)
        freq_predictor = Predictor(reload_freq_model, model_type='frequency')
        freq_predictions = freq_predictor.predict(freq_test)
        freq_predictions_total = freq_predictions.sum()

        ## severity
        
        reload_sev_model = CatBoostRegressor().load_model(sev_model_path)
        sev_predictor = Predictor(reload_sev_model, model_type='severity')
        sev_predictions = sev_predictor.predict(severity_test)


    return None



#%% main
if __name__ == '__main__':
    main()
