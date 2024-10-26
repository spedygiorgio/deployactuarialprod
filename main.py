#%% libraries
import logging
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

#%% main core
def main():
    # load the frequency datasets
    ingestor = Ingestion()
    freq_train, freq_valid, freq_test = ingestor.load_freq()
    logging.info(f'Frequency datasets loaded: train={freq_train.shape}, valid={freq_valid.shape}, test={freq_test.shape}')
    # load the severity datasets
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
    logging.info(f'Frequency model trained: AP ratio={freq_apratio}, MPD={freq_mpd}')
    
    ## severity
    trainer_severity = Trainer(target='severity', train_data=severity_train, val_data=severity_valid, test_data=severity_test, model_type='severity')
    _, _, test_sev_pool = trainer_severity.create_pools()
    severity_model = trainer_severity.train_model()
    sev_apratio, sev_rmse = trainer_severity.evaluate_model(severity_model, test_sev_pool)
    logging.info(f'Severity model trained: AP ratio={sev_apratio}, RMSE={sev_rmse}')
    return None



#%% main
if __name__ == '__main__':
    main()
