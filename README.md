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
```
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
```

---

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt

```

---

## How to Run

This section provides detailed instructions on how to run each script in the project.

### Step 1: Train the Model
Trains the Random Survival Forest and generates the following files:
- `train_predictions.csv`
- `test_predictions.csv`
- `blind_predictions.csv`

To run the script:
```bash
python models/train_rsf_model.py
```

### Step 2: Generate Diagnostic Plots
Creates:
- Permutation feature importance plot
- Predicted vs. Actual scatter plot
- Error histogram
- Kaplan-Meier survival curves

To run the script:
```bash
python analysis/plot_diagnostics.py
```

### Step 3: Estimate Uncertainty
Estimates prediction intervals via bootstrapping.

To run the script:
```bash
python analysis/uncertainty_plot.py
```

### Step 4: Classify Pump Risk Levels
Groups pumps into High, Medium, and Low risk categories.

To run the script:
```bash
python classification/risk_classification.py
```

## Notes
- This project uses **demo placeholders only** for public sharing.
- The original dataset provided by ConocoPhillips is proprietary and cannot be shared publicly.
To replicate this analysis, users must supply their own dataset with a similar structure, including:
    **lifetime_start**     → Start date of pump operation
  
    **lifetime_end**       → End date of pump operation
  
    **FAILSTART (optional)** → Timestamp of failure event (used for uncensored data)
   
    **Numerical features** such as pressure, temperature, vibration, etc.


---

## Author
**Anastasiia Brund**  
Student Researcher | UT Austin

---


