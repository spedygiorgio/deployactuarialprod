#%% Importing libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
#%% Loading the dataset
data_folder = 'data/'
data_file = os.path.join(data_folder, 'french_mtpl.zip')
def main():
    dtypes_list = {'ClaimNb':np.int32, 
    'Exposure':np.float32, 'claims_cost ':np.float32, 'Density':np.int16, 
    'AvgClaimAmount':np.float32, 'BonusMalus':np.int16, 
    'VehPower':np.int16, 'VehAge':np.int16, 'DrivAge':np.int16}


    df = pd.read_csv(data_file, compression='zip',  sep=";", dtype=dtypes_list)#.assign(
        # severity=lambda x: x.apply(lambda row: row['claims_cost'] / row['ClaimNb'] if row['ClaimNb'] > 0 else 0, axis=1)
        #,log_exposure = lambda x: np.log(x['Exposure'])
    #)

    #%% defining predictors and target
    numeric_factors = ['VehPower', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    categorical_factors = ['VehBrand', 'VehGas', 'Region','Area']
    other_variabiles = ['Exposure', 'ClaimNb', 'IDpol', 'claims_cost']


    #%% casting the data types
    for col in categorical_factors:
        df[col] = df[col].astype('category')

    #%% saving the dataset
    train_df = df.query("index <=0.7").filter(items=numeric_factors + categorical_factors + other_variabiles)
    valid_df = df.query("0.7 < index <= 0.8").filter(items=numeric_factors + categorical_factors + other_variabiles)
    test_df  = df.query("0.8 < index").filter(items=numeric_factors + categorical_factors + other_variabiles)

    #%% saving the dataset
    train_df.to_csv(os.path.join(data_folder, 'train.csv'), index=False, sep=";")
    valid_df.to_csv(os.path.join(data_folder, 'valid.csv'), index=False, sep=";")
    test_df.to_csv(os.path.join(data_folder, 'test.csv'), index=False, sep=";")
    print('Dataset saved')
    return None
# %%
if __name__ == '__main__':
    main()