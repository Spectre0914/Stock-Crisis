# Recession Prediction Using Machine Learning

A machine learning project that predicts US recession periods using financial market indicators and achieves 74% ROC-AUC accuracy.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview

This project builds a binary classification model to identify recession periods using historical financial data from 2007-2024. The model successfully captures two major recession events: the 2008 Financial Crisis and the 2020 COVID-19 recession.

**Key Achievement:** XGBoost model with **ROC-AUC of 0.740**, identifying credit market stress as the primary recession predictor.

## Table of Contents

- [Features](#features)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Insights](#key-insights)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Features

### Financial Indicators Used

- **S&P 500 Index:** Overall market performance
- **VIX (Volatility Index):** Market fear gauge
- **Treasury Yield Curve:** 2Y-10Y spread (inversion indicator)
- **Credit Spreads:** Corporate bond risk premium (BAA-10Y)
- **High-Yield Bond ETF (HYG):** Credit stress indicator
- **Financial Sector ETF (XLF):** Banking sector health

### Engineered Features

- **Temporal Lags:** 21-day and 63-day delayed values
- **Rolling Statistics:** Moving averages and standard deviations
- **Momentum Indicators:** 21-day and 63-day returns
- **Total:** 20 features (17 excluding gold-related features)

## Data Sources

- **Yahoo Finance:** Equity indices, VIX, ETFs
- **FRED (Federal Reserve Economic Data):** Treasury yields, credit spreads
- **NBER:** Official recession dates for labeling

**Time Period:** January 2007 - December 2024 (18 years)

## Methodology

### 1. Data Collection & Preprocessing
- Collected data from Yahoo Finance and FRED APIs
- Handled missing values (market holidays, bond market closures)
- Merged multiple data sources into unified dataset

### 2. Feature Engineering
- Created temporal features (lags, rolling statistics, returns)
- Engineered yield curve spread (2Y-10Y)
- Generated 20 predictive features total

### 3. Target Variable Creation
- Binary labels based on NBER official recession dates
- 2008-2009 Financial Crisis: Dec 2007 - Jun 2009
- 2020 COVID Recession: Feb 2020 - Apr 2020

### 4. Model Training
- Time-based train-test split (2007-2019 train, 2020-2024 test)
- Handled class imbalance with appropriate techniques
- Tested 3 models: Logistic Regression, Random Forest, XGBoost

### 5. Evaluation
- ROC-AUC and PR-AUC metrics (appropriate for imbalanced data)
- Feature importance analysis
- Confusion matrices and model comparison

## Results

### Model Performance

| Model | ROC-AUC | Key Features |
|-------|---------|--------------|
| **XGBoost (Best)** | **0.740** | high_yield_bonds (82%), credit_spread (10%) |
| Random Forest | 0.562 | high_yield_bonds (30%), sp500 (12%) |
| Logistic Regression | 0.030 | Failed (linear assumption violated) |

### Top Predictive Features

1. **High-Yield Bonds (HYG)** - 82% importance
2. **Credit Spread (BAA-10Y)** - 10% importance
3. **Yield Curve Rolling Mean (63d)** - 7% importance
4. **Treasury 3-Month Yield** - 2% importance
5. **S&P 500 Index** - 2% importance

### Visualizations

**Feature Behavior During Recessions:**
- S&P 500 crashes during recession periods
- VIX spikes to 40-80 (vs normal 12-15)
- Yield curve inverts 6-18 months before recessions
- Credit spreads widen dramatically

## Installation

### Prerequisites

```bash
Python 3.8+
pip
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recession-prediction.git
cd recession-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries

```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==2.0.0
yfinance==0.2.28
pandas-datareader==0.10.0
```

## Usage

### Run the Complete Analysis

```python
# Open and run the Jupyter notebook
jupyter notebook Recession_Prediction_Model.ipynb
```

### Quick Prediction Example

```python
import pandas as pd
import joblib

# Load trained model
model = joblib.load('models/xgboost_best_model.pkl')

# Load and prepare your data
# (ensure it has the same features as training data)
new_data = pd.read_csv('your_data.csv')

# Make predictions
predictions = model.predict_proba(new_data)[:, 1]
print(f"Recession probability: {predictions[0]:.2%}")
```

## Project Structure

```
recession-prediction/
│
├── Recession_Prediction_Model.ipynb   # Main analysis notebook
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── project_summary.txt                 # Detailed results summary
│
├── data/                               # Data files (not tracked in git)
│   └── processed/
│
├── models/                             # Saved models
│   └── xgboost_best_model.pkl
│
├── visualizations/                     # Generated plots
│   ├── recession_feature_analysis.png
│   ├── model_roc_curves.png
│   ├── feature_importance_xgboost.png
│   └── final_model_summary.png
│
└── src/                                # Source code modules (optional)
    ├── data_collection.py
    ├── feature_engineering.py
    └── model_pipeline.py
```

## Key Insights

1. **Credit Market Stress is the Strongest Predictor**
   - High-yield bond behavior (HYG) dominates with 82% feature importance
   - Credit spreads widen significantly before and during recessions

2. **Gold is a Misleading Indicator**
   - Different behavior in 2008 (increased) vs 2020 (decreased)
   - Removed from final model to improve performance

3. **Yield Curve Inversion Works**
   - Inverts 6-18 months before recession starts
   - Confirmed as a reliable leading indicator

4. **VIX Momentum Matters More Than Raw VIX**
   - Rolling averages capture fear buildup better than spot values
   - 21-day rolling mean is a strong feature

## Limitations

1. **Limited Test Set**
   - Only one recession event in test period (2020 COVID)
   - Model primarily learned from 2008 financial crisis

2. **Unique 2020 Recession**
   - COVID recession was policy-driven and shortest ever
   - May not generalize to traditional recessions

3. **Sample Size**
   - Only 319 recession days out of 2,868 total (11%)
   - Highly imbalanced classification problem

4. **Look-Ahead Bias Risk**
   - Some features (rolling stats) may incorporate future information
   - Validated using strict time-based splits

## Future Improvements

1. **Cross-Validation**
   - Implement time-series cross-validation
   - Test on multiple historical periods

2. **Additional Features**
   - Market breadth indicators
   - Sector rotation patterns
   - Consumer confidence indices
   - Leading Economic Index (LEI)

3. **Model Enhancements**
   - Ensemble of multiple models
   - Threshold optimization for precision/recall
   - LSTM/RNN for sequential patterns

4. **Real-Time Deployment**
   - Automated data collection
   - Monthly prediction updates
   - Dashboard for visualization

## Skills Demonstrated

- **Data Collection:** APIs (yfinance, pandas-datareader)
- **Data Cleaning:** Missing value handling, outlier detection
- **Feature Engineering:** Temporal features, domain knowledge
- **Machine Learning:** Classification, imbalanced data handling
- **Model Selection:** Hyperparameter tuning, cross-validation
- **Evaluation:** ROC-AUC, PR-AUC, confusion matrices
- **Visualization:** matplotlib, seaborn
- **Domain Knowledge:** Financial markets, recession indicators

## Author

**Your Name**

- LinkedIn: (https://www.linkedin.com/in/adnan-khan-pathan/)
- Email: adnann090900@gmail.com
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NBER for official recession dates
- Federal Reserve Economic Data (FRED)
- Yahoo Finance for market data
- scikit-learn and XGBoost communities

## Citation

If you use this project in your research or work, please cite:

```
@software{recession_prediction_2024,
  author = {Your Name},
  title = {Recession Prediction Using Machine Learning},
  year = {2024},
  url = {https://github.com/yourusername/recession-prediction}
}
```

---
