# Rod Pump Failure Analysis

This project applies survival analysis techniques to predict the operational lifetime of rod pumps. It includes data preprocessing, model training using Random Survival Forests, uncertainty estimation via bootstrapping, visualization of results through various diagnostic plots, and classification of pumps into risk categories based on their predicted failure timelines.

This project was developed in academic collaboration with ConocoPhillips, using techniques applied to real-world well failure analysis. The dataset used here is a simulated version for public demonstration only.
---

## Project Overview
- **Modeling**: Trains a Random Survival Forest model
- **Diagnostics**: Evaluates prediction accuracy and model interpretability
- **Uncertainty**: Estimates prediction intervals using bootstrapping
- **Risk Classification**: Groups pumps into High, Medium, and Low risk
---


## Repository Structure
Rod-Pump-Failure-Analysis/
├── analysis/               # Evaluation plots and uncertainty visualization
│   ├── plot_diagnostics.py
│   └── uncertainty_plot.py
├── classification/        # Risk group classification
│   └── risk_classification.py
├── models/                # Model training pipeline
│   └── train_rsf_model.py
├── data/                  # Placeholder demo data
│   ├── train_predictions.csv
│   ├── test_predictions.csv
│   └── blind_predictions.csv
├── requirements.txt       # Python package dependencies
└── README.md              # Project instructions 


## Installation
Install required Python packages:
```bash
pip install -r requirements.txt




