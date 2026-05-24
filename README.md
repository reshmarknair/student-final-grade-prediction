**Student Final Grade Prediction**



This project predicts students' final grade (`G3`) using demographic, behavioral, and academic features.



**Problem Statement**


The goal is to compare two prediction approaches:



 Early-Year Model: uses all features except `G1`, `G2`, and `G3`

 Mid-Year Model: uses all features except `G3`, including `G1` and `G2`


## Project Overview
This project predicts students' final grades (G3) using demographic,
behavioral, and academic data.

Two regression approaches were compared:

- Early-Year Prediction Model
- Mid-Year Prediction Model

The objective is to analyze how prediction accuracy improves when mid-year grades are included.


**Models Used**



 Baseline Model

 Linear Regression

 Random Forest Regressor



**Results**



The best early-year model was Random Forest with an R2 score of 0.31 and MAE of 3.00.



The best mid-year model was Random Forest with an R2 score of 0.80 and MAE of 1.21.



**Conclusion**


Early-year demographic and behavioral features provide limited predictive power. Prediction becomes much more reliable when previous grades `G1` and `G2` are included.

