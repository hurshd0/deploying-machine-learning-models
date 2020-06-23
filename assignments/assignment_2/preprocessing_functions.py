import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    """Loads csv file for training"""
    return pd.read_csv(df_path)


def divide_train_test(df, target):
    """Splits data into train, test distribution"""
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1),  # predictors
        df[target],  # target
        test_size=0.2,  # percentage of obs in test set
        random_state=0
    )  # seed to ensure reproducibility
    return X_train, X_test, y_train, y_test


def add_missing_indicator(df, var):
    """Adds binary indicator for missing values"""
    df[var+'_na'] = np.where(df[var].isnull(), 1, 0)
    return df


def impute_na(df, var, replacement='Missing'):
    """
    Replaces NaN by value entered by user or by string 
    'Missing' as (default behaviour)
    """
    return df[var].fillna(replacement)


def remove_rare_labels(df, var, frequent_labels):
    """
    Groups labels that are not in the frequent list into the umbrella term
    "Rare". 
    """
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')


def encode_categorical(df, var):
    """
    Adds ohe variables and removes original categorical variables
    ohe = one hot encoding
    """
    df = df.copy()
    df = pd.concat(
        [df, pd.get_dummies(df[var], prefix=var, drop_first=True)],
        axis=1
    )
    df.drop(labels=var, axis=1, inplace=True)
    return df


def check_dummy_variables(df, dummy_list):
    """
    Checks that all missing variables where added when encoding, otherwise
    adds the ones that are missing
    """
    for var in dummy_list:
        if var not in df.columns:
            df[var] = 0
    return df


def train_scaler(df, output_path):
    """Trains and saves scaler"""
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df, output_path):
    """Loads the scaler and scales data"""
    scaler = joblib.load(output_path)
    return scaler.transform(df)


def train_model(df, target, output_path):
    """Trains and saves the model
    """
    # initialise the model
    lin_model = LogisticRegression(C=0.0005, random_state=0)

    # train the model
    lin_model.fit(df, target)

    # save the model
    joblib.dump(lin_model, output_path)

    return None


def predict(df, model):
    """Loads model and get predictions"""
    model = joblib.load(model)
    return model.predict(df)


def predict_proba(df, model):
    """Loads model and get predictions probabilties"""
    model = joblib.load(model)
    return model.predict_proba(df)
