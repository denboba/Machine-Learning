import json

# Minimal notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text.split('\n')})

def code(text):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.split('\n')})

# Start building notebook
md("# Tema Învățare Automată - Implementare Completă\\n\\nAceastă temă urmează strict cerințele din tema.pdf")

code("""# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
print("Libraries loaded!")""")

md("## PARTEA 1: Închiriere Biciclete\\n### 4.1 EDA - Exploratory Data Analysis")

code("""# Load data
df_train = pd.read_csv('train_split.csv', parse_dates=['data_ora'])
df_test = pd.read_csv('eval_split.csv', parse_dates=['data_ora'])
print("Data loaded!")
print(df_train.info())""")

with open('tema_ml_complete.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Basic notebook created!")
