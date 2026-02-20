# Design Rules from Interpretable Machine Learning Enable High-Rate CO₂ Electrolysis in Perovskites
This repository contains the machine learning (ML) framework and curated dataset supporting the research article published in *Nature Communications*:  

**Title**: Design Rules from Interpretable Machine Learning Enable High-Rate CO₂ Electrolysis in Perovskites  
**Journal**: Nature Communications  

## Abstract
We develop an interpretable regression workflow to quantitatively uncover structure–property relationships that govern high-rate CO₂ electrolysis performance in perovskite oxides.

## Repository Structure
| File/Directory               | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `Data_preprocessing.ipynb`   | Jupyter notebook for descriptor construction and data preprocessing         |
| `train.py`                   | Python script for supervised regression model training pipeline (note: corrected typo from "trian.py") |
| `data_features.csv`          | Curated dataset with engineered features for ML modeling                    |
| `*.joblib`                   | Saved trained models and preprocessing objects (e.g., StandardScaler)       |
| `model_predictions.csv`      | Independent test set predictions from all trained models                    |

## Computational Environment
### Dependencies
- Python ≥ 3.8
- scikit-learn
- pandas
- numpy
- joblib
