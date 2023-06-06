import yaml

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import os
import json
import warnings

import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
import joblib


def load_draft_data(csv):
    """
    Read data from .csv file into DataFrame
    :param to csv: path to .csv file
    :return DataFrame
    """
    df_draft = pd.read_csv(csv)
    print("Data loaded")
    return df_draft


def add_data_columns(df_big):
    """
    Function that adds additional year, month, and day columns
    filled in (50% of the independent variables)
    # Функція, що додає додаткові колонки року, місяця та дня #
    :param df_big: input DataFrame
    :return: DataFrame
    """
    df_big.Date = pd.to_datetime(df_big[config.get("date_column")])
    df_big['Year'] = df_big[config.get("date_column")].dt.year
    df_big['Month'] = df_big[config.get("date_column")].dt.month
    df_big['Day'] = df_big[config.get("date_column")].dt.day
    print("Additional year, month, and day columns have added.")
    return df_big


def float_data_mean(df_prim):
    """
    Function that Replaces zero non-continuous values with monthly averages
    # Функція, що замінює нульові неперервні значення на середньомісячні #
    :param df_prim: input DataFrame
    :return: DataFrame
    """
    df_prim_nulrow = df_prim.loc[:, df_prim.dtypes == 'float64']  # an array of continuous values
    nul_col_list = df_prim_nulrow.columns[df_prim_nulrow.isnull().any()].tolist()  # a list of columns that have zero
    # non-null values

    # replaces zero continuous values with monthly averages:
    for val in nul_col_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ser_1 = df_prim[val].fillna(df_prim.groupby(config.get('columns_groupby'))[val].transform(lambda x: x.
                                                                                             fillna(x.median())))
            df_prim[val] = ser_1.values

    # removes zero continuous values that have not been replaced by monthly averages:
    for val in nul_col_list:
        df_prim = df_prim.dropna(subset=[val])

    print("Zero non-continuous values have replased with monthly averages.")
    return df_prim


def scaling_floats(df_income):
    """
    Function that scaling float columns
    # Функція, що маштабує неперервні значення в колонках #
    :param df_income: input DataFrame
    :return: DataFrame
    """
    cols_to_scale = df_income.loc[:, df_income.dtypes == 'float64'].columns.tolist()  # list of digital speakers
    # for scaling
    scaler = StandardScaler()
    df_income[cols_to_scale] = scaler.fit_transform(df_income[cols_to_scale])

    print("Float columns have scaled with StandardScaler.")
    return df_income


def categ_data_mode(df_prm):
    """
    Function that replaces null categorical values with Mode
    # Функція, що замінює нульові категоріальні значення на Моду #
    :param df_prm: input DataFrame
    :return: DataFrame
    """
    columns_2 = df_prm.loc[:, df_prm.dtypes != 'float64'].columns.tolist()  # list of categorical data columns
    miss_cat = []  # a list of categorical columns that are missing values

    for val in columns_2:
        if df_prm[val].isnull().mean() != 0.:
            miss_cat.append(val)

    df_prm[miss_cat] = df_prm[miss_cat].astype(str)

    if config.get("missing_cat") == "mode" and miss_cat != []:
        #  replaces null categorical values with Mode
        for val in miss_cat:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ser_1 = df_prm.groupby(["Location", "Year", "Month"])[val].transform(lambda x: x.fillna(get_mode))
                df_prm[val] = ser_1.values

        df_prm = df_prm[~df_prm[miss_cat].isnull().all(axis=1)]  # deletes categorical data rows where all values
    # are missing at the same time
    print("Null categorical values have replaced with monthly mode.")
    return df_prm


def get_mode(x):
    return x.value_counts().index[0]  # mode function for text values


def make_df_csv(pr_df):
    """
    Function that writing the processed dataset to a csv file
    # Функція, що записує записує оброблений набір даних у csv-файл #
    :param pr_df: DataFrame with dictionary for decoding categorical columns
    :return: None
    """
    # Checking the existence of the "results" directory
    # Перевірка існування директорії "results"
    if not os.path.exists("results"):
        os.makedirs("results")

    # Checking the existence of the "data.csv" file and writing to the file
    # Перевірка існування файлу "data.csv" та запис в файл
    if os.path.isfile("results/rain_prediction.csv"):
        pr_df.to_csv("results/rain_prediction.csv", index=False, mode="w")
    else:
        pr_df.to_csv("results/rain_prediction.csv", index=False)
    print("Dataset have just been written to 'results/rain_prediction.csv'.")
    return


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv('new_data.csv')

fgh = len(df)

df = add_data_columns(df)

df = float_data_mean(df)
rty = len(df)
if fgh != rty:
    print("Not all data are filled in!!!")
    fgh = rty

# cols_to_scale = df.loc[:, df.dtypes == 'float64'].columns.tolist()

df = scaling_floats(df)

# df = df.drop(labels=config.get("date_column"), axis=1)
# print("Date column has deleted.")

df[config.get("bool_type_columns")] = df[config.get("bool_type_columns")].replace(["Yes", "No"], [1, 0])
# print('"bool_type_columns" have replased with [1, 0].')

df = categ_data_mode(df)
rty = len(df)
if fgh != rty:
    print("Not all data are filled in!!!")

with open('results/feautures.txt', 'r') as f:
    data = f.read().replace('\\n', '')
col_list = eval(data)

target = {}
predict_mod = {}
if 'RainTomorrow' in col_list:
    target = df.RainTomorrow
    col_list.remove('RainTomorrow')

df1 = df[col_list]

with open('results/key_info.txt', 'r') as f:
    data = f.read().replace('\\n', '')
my_dict: object = eval(data)

columns_3 = df1.loc[:, (df1.dtypes != 'float64') & (~df1.columns.isin(config.get("bool_type_columns")))].columns\
    .tolist()
for val in columns_3:
    if val not in df1.columns.tolist():
        print("The features ", val, " not found in incoming data!!!")

df1 = df1.replace(my_dict)

# load the model from disk
loaded_model = joblib.load('results/finalized_model.sav')

features = df1.loc[:, df1.columns != "RainTomorrow"]

try:
    result = loaded_model.score(features, target)
    print("score =", result)
except:
    print("Target is not exist!")

try:
    predict_mod = loaded_model.predict(features)
except:
    print("Error!!! Not all data are filled in!!!")

try:
    cm = confusion_matrix(target, predict_mod)
    print(cm)
except:
    print("Target is not exist!")


df_predict = df[['Location', 'Date']].copy()
# df_predict = df[['Location', 'Year', 'Month', 'Day']].copy()
# Об'єднання колонок "day", "month" та "year" в одну колонку "date"
# df_predict['Date'] = pd.to_datetime(df_predict[['Year', 'Month', 'Day']])
# Видалення всіх колонок крім "date"
# df_predict.drop(columns=['Year', 'Month', 'Day'], inplace=True)

df_predict["RainTomorrow"] = predict_mod
df_predict['RainTomorrow'] = df_predict['RainTomorrow'].replace([1, 0], ["Yes", "No"])

make_df_csv(df_predict)