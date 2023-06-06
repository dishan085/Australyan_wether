#!/usr/bin/env python
# coding: utf-8

import yaml

import pandas as pd
import numpy as np

import os
import json
import warnings

import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def clean_corr(df_inc, koef_max_cor):    
    """
    Function that deletes correlated columns according to a limit
    # Функція, що видаляє корельовані стовпчики # 
    :param df_inc: input DataFrame
    :param  koef_max_cor: correlation coefficient, above which unnecessary columns are removed
    :return: DataFrame
    """
    k = 0
    end_list = []
    del_columns = [1]
    
    while del_columns != end_list:        
        
        df_corr = df_inc.loc[:, df_inc.dtypes == 'float64'].corr()
    
        # Create a list of values that correlate with each other with a coefficient >= koef_max_cor
        # Створення списку зі значень, що корелюються між собою з коефіцієнтом >= koef_max_cor
    
        corr_list = []
        for column in df_corr:
            corr_vals = df_corr[column][df_corr[column] >= koef_max_cor].index.tolist()
            for val in corr_vals:
                if val != column and (val, column) not in corr_list:
                    corr_list.append((column, val))
                    
        if corr_list == []:
            return df_inc
                    
        new_result1 = list([value[0] for value in corr_list])
        new_result2 = list([value[1] for value in corr_list])
        columns_1 = list(set(new_result1 + new_result2))
                
        # Determine the amount of outliers by column
        # Визначаємо кількість викидів по колонках

        if columns_1 != []:
            quantiles = df_inc.loc[:, df_inc.dtypes == 'float64'].quantile([0.05, 0.95])
            outliers = df_inc[(df_inc[columns_1] < quantiles.loc[0.05, columns_1]) | (df_inc[columns_1] > quantiles.
                                                                                      loc[0.95, columns_1])]
            outliers_count = outliers[columns_1].count()
            
            # Determine the list of correlated columns with the highest emissions
            # Визначаємо список корелюючих колонок, у яких найбільше викидів
    
            del_columns = []
            for i in range(len(new_result1)):                
                a = new_result1[i]
                b = new_result2[i]
                if outliers_count[a] >= outliers_count[b]:
                    del_columns.append(a)
                else:
                    del_columns.append(b)
        
            del_columns = list(set(del_columns))
            
            # Removes repeatedly correlated values from the list of correlated columns
            # Видаляє повторно корелюючі значення зі списку корелюючих колонок
            
            for i in range(len(corr_list)):
                d = list(corr_list[i])
                if all(elem in del_columns for elem in d):
                    a = d[0]
                    b = d[1]
                    if outliers_count[a] >= outliers_count[b]:
                        del_columns.remove(a)
                    else:
                        del_columns.remove(b)
        
        # Delete by column name
        # Видаляємо по іменам колонок
        
        df_inc = df_inc.drop(labels=del_columns, axis=1).copy()
        k = k + 1
                  
    return df_inc


def get_mode(x):
    return x.value_counts().index[0]  # mode function for text values


def load_draft_data(csv):
    """
    Read data from .csv file into DataFrame
    :param to csv: path to .csv file
    :return DataFrame
    """
    df_res = pd.read_csv(config.get("draft_csv"))
    print("Draft data loaded")
    return df_res


def del_empty_columns(df_or, null_limit):    
    """
    Function that deletes columns that are filled less than the limit
    # Функція, що видаляє стовпчики, що заповнені менше від ліміту # 
    :param df_or: input DataFrame
    :param  null_limit: share of empty values in columns, shows which columns will be deleted
    :return: DataFrame
    """   
    # Shows the share of missing values in the columns
    # Показує долю відсутніх значень в колонках
    ser_1 = df_or.isnull().mean() 
    
    # Shows the list of columns to be deleted by the null_limit limit
    # Показує список колонок, що підлягають видаленню по ліміту null_limit    
    del_column_list = ser_1[ser_1 >= null_limit].index.tolist()
    
    # Deletes columns according to the list
    # Видаляє колонки згідно зі списком
    df_or = df_or.drop(labels=del_column_list, axis=1)
    
    print("Columns that are filled less than the limit have deleted:", del_column_list, ", null_limit =", null_limit)
    return df_or 


def del_outliers_rows(df_pr, tail_len):    
    """
    Function that deletes rows with outliers according to the tail_len parameter
    # Функція, що видаляє строки з викидами згідно з параметром tail_len # 
    :param df_pr: input DataFrame
    :param  tail_len: maximum length of the boksplot tail for screening out rows with outliers
    :return: DataFrame
    """ 
    columns1 = df_pr.loc[:, df_pr.dtypes == 'float64'].columns.tolist()
    q1 = df_pr[columns1].quantile(0.25)
    q3 = df_pr[columns1].quantile(0.75)
    iqr = q3 - q1
    df_pr = df_pr[~((df_pr[columns1] < (q1 - tail_len * iqr)) | (df_pr[columns1] > (q3 + tail_len * iqr))).any(axis=1)]
    print("Lines with outliers according to the tail_len parameter have deleted:", ", tail_len =", tail_len)
    return df_pr


def del_nultarget_rows(df_orig):
    """
    Function that removes blank rows of Boolean columns and in which less than n values are 
    filled in (50% of the independent variables)
    # Функція, що видаляє рядки, в яких заповнені менше n значень (50% від незалежних змінних) # 
    :param df_orig: input DataFrame    
    :return: DataFrame
    """
    df_orig = df_orig.dropna(thresh=round(len(df_orig.columns)/2, 0)).reset_index(drop=True)
    df_orig.dropna(subset=config.get("bool_type_columns"), axis=0, inplace=True)  # deletes blank rows by
    # the target column
    # print("Blank rows of Boolean columns and in which less than n values are filled in (50% of the independent"
        # "variables) have deleted.")
    return df_orig


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


def encod_cat_columns(my_df):
    """
    Function that encoding of categorical columns
    # Функція, що показує список колонок категоріальних даних що повинні бути закодовані # 
    :param my_df: input DataFrame    
    :return: DataFrame
    """
    columns_3 = my_df.loc[:, (my_df.dtypes != 'float64') & (~my_df.columns.isin(config.get("bool_type_columns")))]\
        .columns.tolist()
    ordinal_encoder = ce.OrdinalEncoder(cols=columns_3)
    df_copy = ordinal_encoder.fit_transform(my_df)
    # print("Categorical columns have encoded wiht OrdinalEncoder.")
    return df, df_copy


def keyword_dic(inp_df, inp_df_copy):
    # Create a keyword dictionary for decoding categorical columns
    """
    Function that create a keyword dictionary for decoding categorical columns
    # Функція, що створює словник ключових слів для розшифровки категорійних колонок # 
    :param inp_df: input DataFrame
    :param inp_df_copy: input DataFrame with encoding of categorical columns
    :return: key_info: DataFrame with dictionary for decoding categorical columns
    """
    key_info = {}
    columns_3 = inp_df.loc[:, (inp_df.dtypes != 'float64') & (~inp_df.columns.isin(config.get("bool_type_columns")))]\
        .columns.tolist()
    for val in columns_3:    
        name_dict = val
        key_dict = dict(zip(inp_df[val].unique(), inp_df_copy[val].unique()))    
        key_info[name_dict] = key_dict    
    # print(key_info)
    print("Keyword dictionary for decoding categorical columns have created.")
    return key_info


def make_txt_keys(inc_key_info):   
    """
    Function that writing keys for decoding categorical columns to a txt file
    # Функція, що записує ключі для декодування категорійних стовпців у txt-файл #
    :param inc_key_info: dictionary for decoding categorical columns
    :return: None
    """

    # Writing a dictionary to a TXT file
    # Запис словника в TXT файл
    
    # Checking the existence of the "results" directory
    # Перевірка існування директорії "results"
    if not os.path.exists("results"):
        os.makedirs("results")

    # Checking the existence of the file "inc_key_info.txt"
    # Перевірка існування файлу "inc_key_info.txt"
    if os.path.isfile("results/inc_key_info.txt"):
        # Writing a dictionary to a txt file
        # Запис словника в txt файл
        with open("results/key_info.txt", "w") as f:            
            f.write(str(key_info))
    else:
        with open("results/key_info.txt", "w") as f:            
            f.write(str(key_info))
    print("Keys to decode categorical columns to txt file have just been written to 'results/inc_key_info.json'.")
    return


def make_feautures_keys(list_col):
    # Writing keys for decoding categorical columns to a TXT file

    # Запис словника в TXT файл
    # Перевірка існування директорії "results"
    if not os.path.exists("results"):
        os.makedirs("results")

    # Перевірка існування файлу "key_info.txt"
    if os.path.isfile("results/feautures.txt"):
        # Запис словника в TXT файл
        with open("results/feautures.txt", "w") as f:
            f.write(str(list_col))
    else:
        with open("results/feautures.txt", "x") as f:
            f.write(str(list_col))
    print("Feautures list to TXT file have just been written to 'results/feautures.txt'.")
    return


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
    if os.path.isfile("results/data.csv"):
        pr_df.to_csv("results/data.csv", index=False, mode="w")
    else:
        pr_df.to_csv("results/data.csv", index=False)
    print("Dataset have just been written to 'results/data.csv'.")
    return   


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

df1 = load_draft_data(config.get("draft_csv"))

# Create a dictionary in which we record the columns to be deleted in accordance with different values of the
# "korelation_limit" coefficient
# Формуємо словник, в якому записуємо колонки, що треба видалити у відповідності до різних значень коефіцієнту
# "korelation_limit"
clean_corr_col = {}
for korel_limit in config.get("korelation_limit"):
    df2 = clean_corr(df1, korel_limit)
    clean_corr_col[korel_limit] = list(set(df1.columns) - set(df2.columns))

# ----------------------------------------------------------------------------------------------------------------------
# Starting the main cycle of processing a data array depending on the values of the values: "null_limit",
# "korelation_limit", "insert_day", "insert_Rainfall", "tail_len"
# Запуск основного циклу обробки масиву даних в залежності від значень величин: "null_limit", "korelation_limit",
# "insert_day", "insert_Rainfall", "tail_len"
ni = 0
clean_corr = {}

for null_limit in config.get("null_limit"):
    for korelation_limit in config.get("korelation_limit"):
        for insert_day in config.get("insert_day"):
            for insert_Rainfall in config.get("insert_Rainfall"):
                for tail_len in config.get("tail_len"):
                    try:
                        print("Iteration #", ni)
                        df = load_draft_data(config.get("draft_csv"))
                        df = del_empty_columns(df, null_limit)
                    
                        try:
                            # df = clean_corr(df, korelation_limit)
                            col_names = clean_corr_col.get(korelation_limit)
                            for name_col in col_names:                               
                                if name_col in df.columns:                                    
                                    df = df.drop(labels=name_col, axis=1)  # - по іменам колонок
                            
                            # print("Correlated columns according to a limit have deleted:", " korelation_limit =",
                            # korelation_limit)#
                        except:
                            print("Correlated columns according to a limit have not deleted!")

                        df = del_outliers_rows(df, tail_len)
                    
                        df = del_nultarget_rows(df)

                        df = add_data_columns(df)

                        df = float_data_mean(df)

                        df = scaling_floats(df)

                        df = df.drop(labels=config.get("date_column"), axis=1)
                        # print("Date column has deleted.")

                        df[config.get("bool_type_columns")] = df[config.get("bool_type_columns")].\
                            replace(["Yes", "No"], [1, 0])
                        # print('"bool_type_columns" have replaced with [1, 0].')

                        df = categ_data_mode(df)

                        df = df.drop_duplicates()  # shows a table of rows that are not duplicated for all row values

                        df, df_copy = encod_cat_columns(df)                        

                        try:    
                            if config.get(insert_day) == 0:
                                df_copy = df_copy.drop(labels='Day', axis=1)
                                # print("Column Day has deleted.")
                            if config.get(insert_Rainfall) == 0:
                                df_copy = df_copy.drop(labels='Rainfall', axis=1)
                                # print("Column  Raifall has deleted.")
                        except:
                            print("Column Day or Rainfall has non deleted!")
                        
                        df_copy = df_copy.drop(labels='Year', axis=1)
                        # print("Column Year has deleted.")

                        df = df_copy
                    
                        # Split data into train\test for LogisticRegression
                        features = df.loc[:, df.columns != "RainTomorrow"]
                        target = df.RainTomorrow
                        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3,
                                                                            random_state=42)
                    
                        log_reg = LogisticRegression(max_iter=500)
                        model = log_reg.fit(x_train, y_train)
                        predict = log_reg.predict(x_test)
                        accur = accuracy_score(y_test, predict)                    
                        tn, fp, fn, tp = confusion_matrix(y_test, predict).ravel()
                        norma = tp / fn * accur                        
                    
                        # filling in the iterative dataframe
                        new_row = {"null_limit": null_limit, "korelation_limit": korelation_limit,
                                   "missing_cat": config.get("missing_cat"), "insert_day": insert_day,
                                   "insert_Rainfall": insert_Rainfall, "tail_len": tail_len, "insert_day": insert_day,
                                   "len_df": len(df), "norma": norma}
                        if ni == 0:
                            new_row = {"null_limit": null_limit, "korelation_limit": korelation_limit,
                                       "missing_cat": config.get("missing_cat"), "insert_day": insert_day,
                                       "insert_Rainfall": insert_Rainfall, "tail_len":
                                           tail_len, "insert_day": insert_day, "len_df": len(df), "norma": norma}
                            accuracy_df = pd.DataFrame(new_row, index=[0])
                        
                        if ni != 0:
                            accuracy_df = pd.concat([accuracy_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                    except:
                        print("Error of operation! iteration =", ni)
                    print("accuracy_score=", accur, " tp/fn =", tp / fn, " norma =", norma)
                    print("-----------------------------------------------------------------------------------------"
                          "------------------------------------")
                    ni += 1 

# -----------------------------------------------------------------------------------------------------------------------
# Selecting from accuracy_df the dataset with the highest product values "norma"*"len_df"
# Вибір із accuracy_df набору даних з максимальними показниками добутку "norma" * "len_df"
max_index = (accuracy_df["len_df"] * accuracy_df["norma"]).idxmax()
row_max = accuracy_df.loc[max_index].to_dict()

# Writing the iterative dataset to a csv file
# Writing the iterative dataset to a csv file

# Checking the existence of a directory "results"
# Перевірка існування директорії "results"
if not os.path.exists("results"):
    os.makedirs("results")

# Checking the existence of a file "accuracy_prepare.csv"
# Перевірка існування файлу "data.csv"
if os.path.isfile("results/accuracy_prepare.csv"):
    accuracy_df.to_csv("results/accuracy_prepare.csv", index=False, mode="w")
else:
    accuracy_df.to_csv("results/accuracy_prepare.csv", index=False)
print("Dataset have just been written to 'results/accuracy_prepare.csv'.")

# -----------------------------------------------------------------------------------------------------------------------
# Starting the main cycle of processing a data array depending on the values of the values in row_max: 
# Запуск кінцевого циклу обробки масиву даних в залежності від значень величин в row_max: 

df = load_draft_data(config.get("draft_csv"))

df = del_empty_columns(df, row_max.get("null_limit"))
                    
try:    
    korelation_limit = row_max.get('korelation_limit')
    col_names = clean_corr_col.get(korelation_limit)
    for name_col in col_names:                               
        if name_col in df.columns:                                    
            df = df.drop(labels=name_col, axis=1)
    print("Correlated columns according to a limit have deleted:", " korelation_limit =", korelation_limit)
except:
    print("Correlated columns according to a limit have not deleted!")
                    
df = del_outliers_rows(df, row_max.get("tail_len"))
                    
df = del_nultarget_rows(df)

df = add_data_columns(df)

df = float_data_mean(df)

df = scaling_floats(df)

df = df.drop(labels=config.get("date_column"), axis=1)
# print("Date column has deleted.")

df[config.get("bool_type_columns")] = df[config.get("bool_type_columns")].replace(["Yes", "No"], [1, 0])
# print('"bool_type_columns" have replaced with [1, 0].')

df = categ_data_mode(df)

df = df.drop_duplicates()  # shows a table of rows that are not duplicated for all row values

df, df_copy = encod_cat_columns(df)

try:    
    if row_max.get("insert_day") == 0:
        df_copy = df_copy.drop(labels='Day', axis=1)
        df = df.drop(labels='Day', axis=1)
        # print("Column Day has deleted.")
        if row_max.get("insert_Rainfall") == 0:            
            df_copy = df_copy.drop(labels='Rainfall', axis=1)
            df = df.drop(labels='Rainfall', axis=1)
            # print("Column  Raifall has deleted.")
except:
    print("Column Day or Rainfall has non deleted!")
                        
df_copy = df_copy.drop(labels='Year', axis=1)
df = df.drop(labels='Year', axis=1)
# print("Column Year has deleted.")
print("--------------------------------------------------------------------------"
      "---------------------------------------------------")
                    
key_info = keyword_dic(df, df_copy)
make_txt_keys(key_info)
df = df_copy
make_df_csv(df)
col_list = df.columns.tolist()
make_feautures_keys(col_list)

