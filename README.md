**Student Final Grade Prediction**



This project predicts students' final grade (`G3`) using demographic, behavioral, and academic features.


**Project Overview**


This project predicts students' final grades 'G3' using demographic,
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


**Dataset Information**

Dataset Source:
UCI Student Performance Dataset

Target Variable:
- G3 (final grade)

Features:
- study time
- absences
- family support
- previous grades and so on


**Visualization**

images/correlation_heatmap.png

