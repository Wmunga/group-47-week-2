Group 47

# Climate Data ML Project (SDG 13: Climate Action)

## Overview
This project leverages machine learning to analyze and predict yearly average temperature changes, directly supporting the United Nations Sustainable Development Goal 13: Climate Action. By forecasting climate trends, the project aims to raise awareness and inform decision-making for a more sustainable future.

## How It Works
- Loads and preprocesses historical climate data from a CSV file.
- Trains a linear regression model to predict future yearly average temperatures based on past trends.
- Evaluates model performance using Mean Absolute Error (MAE) and R² score.
- Visualizes actual vs. predicted temperature trends.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

## Usage
1. Place `data.csv` in the project folder.
2. Run the script:
   ```
   python data_processor.py
   ```
3. View the output in your terminal and the prediction plot.

## Results
- The model outputs MAE and R² score.
- A plot displays predicted vs. actual yearly average temperatures.

## Ethical Reflection
Potential bias may exist in the data due to incomplete or non-representative samples. The model is for educational purposes and should not be used for policy decisions without expert review. By making climate trends more understandable, this project supports SDG 13 (Climate Action) and promotes fairness through open data and transparent methods

## Screenshots
![Prediction Plot](prediction_plot.png)


## 1-Page Project Summary

**SDG Problem Addressed:**  
Climate change (SDG 13: Climate Action) — Predicting yearly average temperature changes to inform and raise awareness about global warming trends.

**ML Approach Used:**  
Supervised learning with linear regression to forecast future temperature trends based on historical data.

**Results:**  
- Model achieved MAE of 0.0516 and R² 0.9271.
- Visualization shows predicted vs. actual temperature trends, highlighting the model’s ability to capture climate patterns.

**Ethical Considerations:**  
- Data bias may affect predictions if the dataset is incomplete or not globally representative.
- The model is transparent and uses open data.
- Promotes climate awareness and supports sustainability by making climate data accessible and understandable.

---
