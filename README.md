# Auto Insurance Pricing Model

## Overview
This project implements a solution for the auto insurance pricing challenge described in `data/assignment.pdf`. The goal is to develop a competitive pricing model that balances market share (at least 30%) while maximising average margin with respect to competitors' pricing.

## Model Architecture

### Core Components
1. **Feature Processing**
   - Basic feature engineering class:
        - temporal features
        - categorical features
        - filling of NaN values.

2. **Pricing Model**
   - Multi-quantile XGBoost regression.
   - Multi-Objective Genetic Algorithm optimisation.

3. **Notebooks**
    - Main prediction pipeline: `notebooks/pipeline.ipynb`
    - Exploratory notebooks
    - Dimensionality reduction test 


## Project Structure
```
auto-insurance/
├── data/
│   ├── assignment.pdf
│   ├── train.xlsx
│   └── test.xlsx
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── exploratory_analysis_2.ipynb
│   ├── pipeline.ipynb
│   └── dim_reduction.ipynb
└── src/
    ├── feature_processor.py
    ├── model.py
    └── utils.py
```
