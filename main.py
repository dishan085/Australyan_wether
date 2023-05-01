#!/usr/bin/env python
# coding: utf-8

import yaml
import logging
import subprocess

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

import os

import category_encoders as ce
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, \
    roc_auc_score, f1_score, classification_report

from warnings import simplefilter


def load_data(csv):
    """
    Read data from .csv file into DataFrame
    :param to csv: path to .csv file
    :return DataFrame
    """
    df_draft = pd.read_csv(csv)
    print("Data loaded")
    return df_draft        


def smote_split(df_pr):  # Cross-validation
    """
    Splitting data in train/test 
    :param df_pr: input DataFrame
    :return: train and test data
    """
    
    x_train_pr, x_test_pr, y_train_pr, y_test_pr = train_test_split(df_pr.loc[:, df_pr.columns != config.get("target")], 
                            df_pr[config.get("target")], test_size=config.get("test_size"), random_state=42)

    # Balancing classes in training data with SMOTE
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_pr, y_train_pr)
    print("Data split.")
    return x_train_resampled, x_test_pr, y_train_resampled, y_test_pr


def tune_models(x_train_inc, y_train_inc):
    """
    Tuning the hyper-parameters of estimators and selecting the best estimators
    :param  x_train_inc:  train data  - features
    :param  y_train_inc: train data  - target 
    :return:   dictionary of best estimators
    """
    
    #models = {"log_reg": LogisticRegression()}
    models = {"random_forest" : RandomForestClassifier()}
    #models = {"log_reg": LogisticRegression(), "knn": KNeighborsClassifier(),
              #"random_forest" : RandomForestClassifier()}
    #models = {"knn": KNeighborsClassifier(),
              #"random.forest" : RandomForestClassifier()}    
    best_estimators = {} 
    logging.info("TUNING MODELS...") 
    for estimator_name, estimator in models.items():
        clf = GridSearchCV(estimator=estimator, param_grid=config.get(estimator_name),
                 scoring="accuracy", cv=10)
        clf.fit(x_train_inc, y_train_inc)
        best_score = clf.best_score_
        best_estimators[estimator_name] = clf.best_estimator_
        logging.info("{} best score: {}".format(estimator, best_score)) 
    print("Models tuned.") 
    return best_estimators


def evaluate_models(estimators, x_train_or, x_test_or, y_train_or, y_test_or):
    """
    Retrain best estimators and evaluate them
    :param estimators: dictionary of estimators
    :param x_train_or: train data - features
    :param x_test_or: test data - features
    :param y_train_or: train data - target
    :param y_test_or: test data - target
    :return:
    """
    evaluation_data = {}
    logging.info("EVALUATION: \n")
    for model_name, estimator in estimators.items():
        estimator.fit(x_train_or, y_train_or)
        predict = estimator.predict(x_test_or)
        predict_probability = estimator.predict_proba(x_test_or)[:, 1]
        
        logging.info("{}: \n".format(model_name))
        accuracy = accuracy_score(y_test_or, predict)
        precision = precision_score(y_test_or, predict)
        recall = recall_score(y_test_or, predict)
        cm = confusion_matrix(y_test_or, predict)
        cr = classification_report(y_test_or, predict)
        f1 = f1_score(y_test_or, predict)
        roc_auc = roc_auc_score(y_test_or, predict_probability)
        logging.info("Accuracy: {}".format(accuracy))
        logging.info("Precision: {}".format(precision))
        logging.info("Recall: {}".format(recall))
        logging.info("Confusion matrix: {}".format(cm))
        logging.info("Classification report: {}".format(cr))
        logging.info("F1 score: {}".format(f1))
        logging.info("RIC AUC score: {}".format(roc_auc))
        evaluation_data[model_name] = [accuracy, f1, roc_auc]
        
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', linewidths=5, cbar=False,
                   annot_kws={'fontsize': 15}, yticklabels=['0', '1'], xticklabels=['Predict 0', 'Predict 1'])
        fig.savefig("results/{}_confusion_matrix.png".format(model_name))
    
    df_evaluation = pd.DataFrame.from_dict(evaluation_data, orient='index', 
                                          columns=['accuraccy', 'f1_score', 'roc_auc_score'])
    df_evaluation.to_csv("results/evaluation.csv")       
        
    return


simplefilter(action='ignore', category=FutureWarning)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename="results/logfile")

# Запуск файла обробки дата-сету
#files = ['prepare_dataset.py']  # файлы, которые нужно запустить
#for file in files:
    #subprocess.Popen(args=["start", "python", file], shell=True, stdout=subprocess.PIPE)

df = load_data(config.get("csv"))

x_train, x_test, y_train, y_test = smote_split(df)

best_models = tune_models(x_train, y_train)

evaluate_models(best_models, x_train, x_test, y_train, y_test)
