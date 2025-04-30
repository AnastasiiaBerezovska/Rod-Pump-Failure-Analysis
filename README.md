# Rod Pump Failure Analysis

This project applies survival analysis techniques to predict the operational lifetime of rod pumps. It includes data preprocessing, model training using Random Survival Forests, uncertainty estimation via bootstrapping, visualization of results through various diagnostic plots, and classification of pumps into risk categories based on their predicted failure timelines.

This project was developed in academic collaboration with ConocoPhillips, using techniques applied to real-world well failure analysis. The dataset used here is a simulated version for public demonstration only.

## Project Overview
- **Modeling**: Trains a Random Survival Forest model
- **Diagnostics**: Evaluates prediction accuracy and model interpretability
- **Uncertainty**: Estimates prediction intervals using bootstrapping
- **Risk Classification**: Groups pumps into High, Medium, and Low risk
---

![image](https://github.com/user-attachments/assets/ae798317-5c7b-4f2d-bbcd-ea4402c97588)


---

## Installation

Install required Python packages:
```bash
pip install -r requirements.txt

Train the Model
Trains the Random Survival Forest and generates:

train_predictions.csv

test_predictions.csv

blind_predictions.csv

python models/train_rsf_model.py

Notes
This project uses demo placeholders only for public sharing.
The original dataset from ConocoPhillips is private. Users must supply their own dataset with a similar structure, consisting of:      
    Start date (lifetime_start)
    End date (lifetime_end)
    Failure event (FAILSTART, optional if censored)
    Numerical features such as pressure, temperature, and vibration.






