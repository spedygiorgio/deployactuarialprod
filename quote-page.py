#%% load key apps
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
from typing import Literal
from steps.predict import Predictor
import yaml

#%% load the config file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

#%% funzioni per caricare il predittore CatBoost
@st.cache_resource
def load_scorer(model_path, model_type):
    loaded_model = CatBoostRegressor().load_model(model_path)
    scorer = Predictor(loaded_model, model_type)
    return scorer

#%% define the input schema
def create_input_form():# -> dict[str, Any]:
    VehPower = st.number_input('Vehicle Power', min_value=1, max_value=20, value=1)
    VehAge = st.number_input('Vehicle Age', min_value=0, max_value=120, value=0)
    DrivAge = st.number_input('Driver Age', min_value=18, max_value=120, value=18)
    Density = st.number_input('Density', min_value=0, max_value=30000, value=0)
    BonusMalus = st.number_input('Bonus Malus', min_value=50, max_value=230, value=50)
    VehBrand = st.selectbox('Vehicle Brand', ['B12', 'B3', 'B2', 'B5', 'B4', 'B6', 'B10', 'B1', 'B13', 'B11', 'B14'])
    VehGas = st.selectbox('Vehicle Gas', ['Regular', 'Diesel'])
    Region = st.selectbox('Region', ['R72', 'R91', 'R52', 'R11', 'R94', 'R93', 'R31', 'R82', 'R22', 'R21', 'R42','R54', 'R73', 'R41', 'R26', 'R25', 'R24', 'R53', 'R83', 'R23', 'R74', 'R43'])
    Area = st.selectbox('Area', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    policyholder = {
        'VehPower': VehPower,
        'VehAge': VehAge,
        'DrivAge': DrivAge,
        'Density': Density,
        'BonusMalus': BonusMalus,
        'VehBrand': VehBrand,
        'VehGas': VehGas,
        'Region': Region,
        'Area': Area
    }
    return policyholder

def create_expense_form():
    FixedExpenses = st.number_input('Fixed Expenses', min_value=0.0, value=25.0)
    VariableExpenses = st.number_input('Variable Expenses', min_value=0.0, max_value=0.999, value=0.2)
    TaxRate = st.number_input('Tax Rate', min_value=0.0, max_value=1.0, value=0.05)
    expenses = {
        'FixedExpenses': FixedExpenses,
        'VariableExpenses': VariableExpenses,
        'TaxRate': TaxRate
    }
    return expenses

def compute_commercial_premium(pure_premium, expenses)-> float:
    numerator = pure_premium + expenses.get('FixedExpenses', 0)
    denominator  = 1 - expenses.get('VariableExpenses', 0) - expenses.get('TaxRate', 0)
    commercial_premium = numerator / denominator

    return commercial_premium

#%% calcola il premio puro
def calculate_premium(policyholder, model_freq, model_sev):
    df_insured = pd.DataFrame([policyholder])
    # adds the log exposure = 0 and fake ClaimNb = 1
    df_insured['log_exposure'] = 0.0
    df_insured['ClaimNb'] = 1
    pred_freq = model_freq.predict(df_insured)
    pred_sev = model_sev.predict(df_insured)
    pred_pp = pred_freq[0] * pred_sev[0]
    return pred_freq[0], pred_sev[0], pred_pp

#%% inizializza il session state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

def on_click_callback():
    st.session_state.button_clicked = True
    return None

#%% main
def main():
    st.title('Insurance Premium Calculator')

    # load the models
    model_freq = load_scorer(config['models']['frequency'], 'frequency')
    model_sev = load_scorer(config['models']['severity'], 'severity')

    # load the input form for policyholder data
    st.write('Insert the policyholder data:')
    policyholder = create_input_form()

    st.write('Policyholder data:')
    st.write(policyholder)

    st.write('Insert the expenses data:')
    expenses = create_expense_form()

    # add a button to calculate the premium
    st.button('Calculate premium', on_click=on_click_callback)

    # calculate the pure premium
    if st.session_state.button_clicked is True:
        with st.spinner('Calculating the premium...'):
            frequency, severity, pure_premium = calculate_premium(policyholder, model_freq, model_sev)
            st.write(f'Frequency: {frequency:.3%}')
            st.write(f'Severity: €{severity:,.2f}')
            st.write(f'Pure Premium: €{pure_premium:,.2f}')
        commercial_premium = compute_commercial_premium(pure_premium, expenses)
        st.write(f'Commercial Premium: €{commercial_premium:,.2f}')
        st.session_state.button_clicked = False

    return None

if __name__ == '__main__':
    main()