# Automation of preparing data before training, training on multiple models, and automating predictions from the Kaggle dataset titled Rain in Australia 

## Problem Statement

Preparing datasets before training prediction models is usually a semi-automated process that requires many iterations before obtaining a finished feature space. This project was developed by me to verify and test the automatic processing (the file **"prepare_dataset.py"**) of the input raw data (**"weatherAUS.csv"**) according to the parameters specified in a separate file **"config.yaml"**. The project also developed a code for training several forecasting models (**"main.py"**) and a code for automatic prediction using new input data (**"rain_prediction.py"**). 
This project was developed on the basis of a dataset of weather observations in Australia. The dependent feature in these observations is the RainTomorrow feature, which takes the value **Yes/No**. This is a binary classification task.

## Preparing datasets

Preparation of the dataset includes:
- deleting columns in which the data is less than a certain proportion (the **"null_limit"** parameter in the **"config.yaml"** file)
- deletion of rows that are less than **50%** filled;
- deleting columns that are correlated with other columns by a factor greater than a certain fraction (the **"korelation_limit"** parameter in the **"config.yaml"** file). Moreover, those columns that have more data outliers in the quantile **<0.05 and >0.95** are removed from the correlated pair of columns;
- deletion of rows that fall into certain outliers (parameter **"tail_len"** in the file **"config.yaml"**");
- replacing categorical data gaps with a **mode**;
- replacement of continuous data outliers with **monthly averages**;
- scaling continuous data using **StandardScaler**;
- categorical data encoding using **OrdinalEncoder**.

All the parameters in the configuration file **"config.yaml"** are specified by lists, according to which the iterations of data set preparation take place.

The quality of data preparation is evaluated by training the processed dataset on a **Logistic Regression model** (without Cost-function balancing), which gives a special metric of the data-based error matrix: **norma = accuracy x TP/FN** multiplied by the number of rows of the processed dataset (**len_df**). 
*Note: the norma metric is taken for a specific data set because it is imbalanced on the target feature.*

The results of the dataset preparation iterations are written to the file **"results/accuracy_prepare.csv"**. The best data set ready for training is saved to the file **"results/data.csv"**

The dataset preparation is started by running the file **"prepare_dataset.py"**, or together with the file **"main.py"**, after removing the comments from the code:

```python
files = ['prepare_dataset.py']
for file in files:
    subprocess.Popen(args=["start", "python", file], shell=True, stdout=subprocess.PIPE)
```

## Training of several prediction models

The list of mathematical methods for training the model based on the prepared data set is described in the **"tune_models"** function of the **"main.py"** file in the **"models"** variable.
Parameters of mathematical methods for model training are specified by lists in **"config.yaml"**, according to which iterations are performed.

The training results are written to the following files:
- **results/evaluation.csv** - a table of the best iterations for all models, indicating the values of metrics;
- **results/logfile**;
- **results/model_name_confusion_matrix.png** - pictures of error matrices;
- **results/finalized_model.sav** - the file of the best prediction model;
- **results/model_name_model.sav** - files of prediction models for specific methods.

Training of several prediction models is started by running the **"main.py"** file.

## Automatic forecasting based on new input data

Prediction is performed by loading the raw data file **"new_data.csv"** with the latest observations from the root directory. Moreover, the names of the data features must match the names of the data in the draft set (**"weatherAUS.csv"**).

Prediction is started by running the **"rain_prediction.py"** file.

The prediction result is written to the file **"rain_prediction.csv"**.

## Project Folder Structure:

```text
Australyan_rain 
├──   configs
|     └──  config.yaml
├──  results
|     └── accuracy_prepare.csv
|     └── data.csv
|     └── evaluation.csv
|     └── feautures.txt
|     └── finalized_model.sav
|     └── key_info.txt
|     └── knn_confusion_matrix.png
|     └── knn_model.sav
|     └── log_reg_confusion_matrix.png
|     └── log_reg_model.sav
|     └── logfile
|     └── rain_prediction.csv
|     └── random_forest_confusion_matrix.png
|     └── random_forest_model.sav
|     └── svm_confusion_matrix.png
|     └── svm_model.sav
|     └── tree_confusion_matrix.png
|     └── tree_model.sav
└──  main.py
└──  new_data.csv
└──  prepare_dataset.ipynb
└──  prepare_dataset.py
└──  rain_prediction.py
└──  weatherAUS.csv
```
