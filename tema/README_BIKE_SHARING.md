# Bike Sharing Prediction - Assignment 1

## Overview

Machine learning project for predicting bicycle rental demand using weather conditions, seasonality, and temporal features. Implements Gradient Boosting Regressor with quantile regression for interval predictions.

---

## Dataset

### Files
- **train_split.csv** - Training dataset (6,878 records)
- **eval_split.csv** - Evaluation dataset (4,008 records)

### Features (12 variables)
- **Temporal:** `data_ora` (datetime)
- **Seasonal:** `sezon` (season), `sarbatoare` (holiday), `zi_lucratoare` (working day)
- **Weather:** `vreme` (weather), `temperatura` (temperature), `temperatura_resimtita` (feels-like temp), `umiditate` (humidity), `viteza_vant` (wind speed)
- **Users:** `ocazionali` (casual), `inregistrati` (registered)
- **Target:** `total` (total rentals)

### Time Period
- Training: Jan 1, 2011 - Dec 12, 2012
- Evaluation: Jan 13, 2011 - Dec 19, 2012

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn

### 2. Run Notebook
```bash
jupyter notebook tema1_abdulkadir\ gobena-denboba.ipynb
```

### 3. Expected Output
- Exploratory Data Analysis visualizations
- Model training with RMSE metrics
- `predictii_biciclete.csv` with predictions (low, median, high)

---

## Methodology

### Feature Engineering
Created 4 temporal features:
- `ora` - Hour of day (0-23)
- `zi_saptamana` - Day of week (0-6)
- `luna` - Month (1-12)
- `este_weekend` - Weekend indicator (0/1)

### Model
**Gradient Boosting Regressor** with quantile regression:
- 3 models for interval predictions (5%, 50%, 95% quantiles)
- 100 estimators, learning rate 0.1, max depth 3

### Performance
- **Test RMSE:** ~97 rentals
- **Relative Error:** ~49% (reasonable for high-variability data)

---

## Project Structure

```
tema/
├── train_split.csv                    # Training data
├── eval_split.csv                     # Evaluation data
├── tema1_abdulkadir gobena-denboba.ipynb  # Main notebook
├── Technical_Report_Bike_Sharing.md  # Full technical report
├── README_BIKE_SHARING.md            # This file
└── predictii_biciclete.csv           # Generated predictions
```

---

## Key Results

### Dataset Statistics
- **Training:** 188.87 avg rentals/hour (min: 1, max: 977)
- **Evaluation:** 196.22 avg rentals/hour (min: 1, max: 943)
- **No missing values** in either dataset

### Model Insights
- Strong seasonal and hourly patterns captured
- Weather conditions significantly impact predictions
- Quantile predictions provide uncertainty estimates

---

## Documentation

For detailed analysis, methodology, and results, see:
**Technical_Report_Bike_Sharing.md**

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

**Student:** Abdulkadir Gobena-Denboba  
**Date:** November 2024  
**Status:** Complete and tested
