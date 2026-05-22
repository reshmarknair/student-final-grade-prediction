#!/usr/bin/env python
# coding: utf-8

# ## *Student Final Grade Prediction*
# 
# This project predicts stuntss' final grade (`G3`) using demographic, behavioral, and academic features. Two models are compared: an ear_y-year model without previous grades and a m_d-year model including previous grades.

# In[7]:


import pandas as pd


# In[8]:


df = pd.read_csv("C:/Project/Std_performance_project/Data/student_data.csv",sep=";")


# ### ****Problem Statement****
# 
# The goal of this project is to predict students' final grade (`G3`) and understand which factors contribute most to student performance.
# 
# Two prediction scenarios are considered:
# 
# 1. Early-Year Model: predicts `G3` without using `G1` and `G2`.
# 2. Mid-Year Model: predicts `G3` using all available features, including `G1` and `G2`.
# 
# This comparison helps understand whether demographic and behavioral factors alone are enough, or whether previous academic performance is necessary for accurate prediction.

# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.columns


# In[14]:


for col in df.select_dtypes(include="object").columns: 
    print(col, df[col].unique())


# ## Dataset Overview
# 
# The dataset contains student demographic, social, academic, and lifestyle-related information. The target variable is `G3`, which represents the final grade.
# 
# Important variables include:
# 
# - `G1`: first period grade
# - `G2`: second period grade
# - `G3`: final grade
# - `studytime`: weekly study time
# - `failures`: number of previous class failures
# - `absences`: number of school absences
# - `Medu`: mother's education level
# - `Fedu`: father's education level
# - `Dalc`: weekday alcohol consumption
# - `Walc`: weekend alcohol consumption

# In[15]:


df.isnull().sum()


# In[16]:


df.duplicated().sum()


# In[17]:


df["G3"].describe()


# In[18]:


df['G3'].value_counts()

##helps to understand the distribution of target variable.
##Your target values are not evenly distributed.

#Example:

#10 → 56 samples
#20 → 1 sample

#This is called imbalanced data.

#Machine learning models may become biased toward frequent classes
#Rare classes may be predicted poorly

##Detect unusual or suspicious values ex, score 25 appears for 5 students but our score range is only from 0-20


# ## Explanatory Data Analysis
# 

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.xlabel("Final Grade: G3")
plt.ylabel("Count")
plt.title("Final Grade Distribution")
sns.histplot(df["G3"],bins=20,kde=1)
plt.show()
# appears to be a -ve skewed distribution as tail towards the left and more data points on the right side.
#This indicates that a larger proportion of students achieved higher scores while relatively few scored very low marks.


# In[20]:


# Now lets try to find if there's any relation between the absence of students and their grades

sns.scatterplot(x="absences",y="G3", data = df, alpha=0.5)
plt.title("Absences v/s Final Grades")


#students with fewer absences seem to include many average-to-excellent marks, but the relationship does not look very strong.


# In[21]:


print(df[['absences', 'G3']].corr())


# In[22]:


## no strong relationship between absences and final grade.


# In[23]:


sns.scatterplot(x="studytime",y="G3",data=df)
plt.title("Study Time V/s G3")

plt.show()

sns.lineplot(x="studytime",y="G3",data=df)
plt.title("Study Time V/s G3")

plt.show()


sns.boxplot(x="studytime",y="G3",data=df)
plt.title("Study Time V/s G3")
plt.show()


# In[24]:


print(df[['studytime', 'G3']].corr())


# Positive correlation which means when study time increases grades also increases gradually.
# But in the graph its visible that after a purticular point (3) the graph or the plot a slight decrease which can be interpretted as a result of burnout...

# In[25]:


df.groupby("studytime")["G3"].mean()


# In[26]:


df.groupby('studytime')['G3'].mean().plot(kind='line')
plt.title("Average Final Grade by Study Time")
plt.ylabel("Average G3")
plt.show()


# In[27]:


sns.scatterplot(x="G1", y="G3", data =df)
plt.title("G1 vs Final Grade")
plt.show()

sns.scatterplot(x="G2", y="G3", data =df)
plt.title("G2 vs Final Grade")
plt.show()


# In[28]:


print(df[['G1', 'G3']].corr())
print(df[['G2', 'G3']].corr())


# strong positive correlation

# In[29]:


sns.barplot(x="failures", y="G3", data =df)
plt.title("failures vs Final Grade")
plt.show()


# In[30]:


print(df[["failures","G3"]].corr())


# Less failures more final scores!!!
# Negatively correlated

# In[31]:


sns.boxplot(x="internet",y="G3",data=df)
plt.title("Internet usage V/s G3 grades")


# ## EDA Summary
# 
# The final grade distribution shows that many students scored around the average-to-good range, while fewer students received very low or very high marks.
# 
# The scatterplot between absences and final grade did not show a strong relationship. Similarly, study time showed only a weak relationship with final grade. Previous failures appeared to have a clearer negative relationship with final performance.

# In[32]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(numeric_only=True),annot = True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# ## Correlation Analysis
# 
# The correlation heatmap shows that `G1` and `G2` have the strongest positive correlation with `G3`.
# 
# - `G1` and `G3`: strong positive relationship
# - `G2` and `G3`: very strong positive relationship
# - `failures` and `G3`: moderate negative relationship
# - `studytime`, `absences`, and most demographic variables: weak relationship with `G3`
# 
# This suggests that previous academic performance is likely to be a much stronger predictor of final grade than demographic or behavioral features alone.

# ## Modeling Approach
# 
# This project uses supervised machine learning because the target variable `G3` is already known in the dataset.
# 
# Since `G3` is a numeric value, this is a regression problem.
# 
# Two modeling approaches are used:
# 
# 1. Early-Year Model:
#    - Uses all features except `G1`, `G2`, and `G3`.
#    - Simulates prediction before previous term grades are available.
# 
# 2. Mid-Year Model:
#    - Uses all features except `G3`.
#    - Includes `G1` and `G2`.
#    - Simulates prediction after previous term grades are available.
# 
# The models are evaluated using MAE, RMSE, and R2 score.

# In[33]:


#Import ML Tools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# In[34]:


# Creating Targets and Feature Sets

y = df["G3"]
X_early = df.drop(columns = ["G1","G2","G3"])
X_mid = df.drop(columns =["G3"])


# In[35]:


#Split Data into Tain and Test

#Model 1
X_train_early,X_test_early,y_train,y_test = train_test_split(X_early,y,test_size = 0.2, random_state= 42) 

#Model 2
X_train_mid,X_test_mid,_,_ = train_test_split(X_mid,y,test_size = 0.2, random_state= 42) 


# In[36]:


# Preprocessing Function 

def create_preprocessor(X): 
    numeric_columns = X.select_dtypes(include = ["int64","float64"] ).columns
    categorical_columns = X.select_dtypes(include = ["object"] ).columns
    preprocessor = ColumnTransformer(
        transformers = [("num", StandardScaler(),numeric_columns), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)]
    )
    return preprocessor


# In[37]:


# function evaluation

def evaluate_model(model,X_train,X_test,y_train,y_test,model_name):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mae =mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    print(model_name )
    print("MAE:",round(mae,2) )
    print("RMSE:",round(rmse,2) )
    print("R2:",round(r2,2) )
    print("-"*40)


# ## Evaluation Metrics
# 
# MAE, RMSE, and R2 score are used to evaluate the models.
# 
# - MAE measures the average prediction error in grade points.
# - RMSE also measures prediction error, but gives more penalty to larger mistakes.
# - R2 score shows how much variation in `G3` is explained by the model.
# 
# Lower MAE and RMSE are better. Higher R2 score is better.

# In[38]:


print("Target variable shape:", y.shape)

print("Early-year feature shape:", X_early.shape)
print("Mid model feature shape:", X_mid.shape)

print("Early train shape:", X_train_early.shape)
print("Early test shape:", X_test_early.shape)

print("Mid train shape:", X_train_mid.shape)
print("Mid test shape:", X_test_mid.shape)


# In[39]:


#Builing Model 1 [Early Year Model]

## create preprocessing tool for model 1

early_preprocessor = create_preprocessor(X_early)

early_baseline_model = Pipeline(steps = [
    ("preprocessor", early_preprocessor),
    ("model", DummyRegressor(strategy = "mean"))
])

early_linear_model =  Pipeline(steps = [
    ("preprocessor", early_preprocessor),
    ("model", LinearRegression())
])

early_rf_model =  Pipeline(steps = [
    ("preprocessor", early_preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])


## evaluating model 1

evaluate_model (early_baseline_model,X_train_early,X_test_early,y_train,y_test,"Early_Year Baseline Model")

evaluate_model (early_linear_model,X_train_early,X_test_early,y_train,y_test,"Early_Year Linear Model")

evaluate_model (early_rf_model,X_train_early,X_test_early,y_train,y_test,"Early_Year RF Model")


# *Early-year demographic, behavioral, and school-related features contain some useful signal, but they are not enough for highly accurate prediction.*

# In[40]:


#Builing Model 2 [Mid Year Model]


## create preprocessing tool for model 2

mid_preprocessor = create_preprocessor(X_mid)

mid_baseline_model = Pipeline(steps = [
    ("preprocessor", mid_preprocessor),
    ("model", DummyRegressor(strategy = "mean"))
])

mid_linear_model =  Pipeline(steps = [
    ("preprocessor", mid_preprocessor),
    ("model", LinearRegression())
])

mid_rf_model =  Pipeline(steps = [
    ("preprocessor", mid_preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])


## evaluating model 1

evaluate_model (mid_baseline_model,X_train_mid,X_test_mid,y_train,y_test,"Mid_Year Baseline Model")

evaluate_model (mid_linear_model,X_train_mid,X_test_mid,y_train,y_test,"Mid_Year Linear Model")

evaluate_model (mid_rf_model,X_train_mid,X_test_mid,y_train,y_test,"Mid_Year RF Model")


# *The mid-year models performed much better than the early-year models. The Random Forest model improved from an R2 score of 0.31 in the early-year model to 0.80 in the mid-year model. The MAE also decreased from 3.00 to 1.21. This shows that including G1 and G2 greatly improves prediction accuracy. Therefore, previous academic performance is the strongest predictor of final grade.*

# In[41]:


#Model comparison Table
    

model_results = pd.DataFrame({
    "Model": [
        "Early-Year Baseline",
        "Early-Year Linear Regression",
        "Early-Year Random Forest",
        "Mid-Year Baseline",
        "Mid-Year Linear Regression",
        "Mid-Year Random Forest"
    ],
    "MAE": [3.65, 3.39, 3.00, 3.65, 1.65, 1.21],
    "RMSE": [4.55, 4.19, 3.77, 4.55, 2.38, 2.01],
    "R2 Score": [-0.01, 0.14, 0.31, -0.01, 0.72, 0.80]
})

model_results


# ## Model Results Interpretation
# 
# The early-year models had limited predictive performance. The best early-year model was Random Forest, with an MAE of 3.00 and an R2 score of 0.31. This shows that demographic, social, and behavioral features provide some useful information, but they are not enough for highly accurate grade prediction.
# 
# The mid-year models performed much better. The Mid-Year Random Forest model achieved an MAE of 1.21 and an R2 score of 0.80. This shows that including `G1` and `G2` greatly improves prediction accuracy.
# 
# Overall, previous academic performance is the strongest predictor of final grade.

# In[42]:


# Model Comparison by barcharts

plt.figure(figsize=(15, 6))
sns.barplot(x="Model", y="R2 Score", data=model_results)
plt.xticks(rotation=0)
plt.title("Model Comparison by R2 Score")
plt.show()


# In[43]:


early_rf_predictions = early_rf_model.predict(X_test_early)
mid_rf_predictions = mid_rf_model.predict(X_test_mid)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=early_rf_predictions)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Early-Year Random Forest: Actual vs Predicted G3")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=mid_rf_predictions)
plt.plot([0, 20], [0, 20], color="red", linestyle="--")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Mid-Year Random Forest: Actual vs Predicted G3")
plt.show()


# The mid-year model predictions are closer to the red reference line, meaning they are closer to the actual values. This confirms that the mid-year model performs better than the early-year model.

# **At Risk Student Identification**
# 
#  If G3<10, the student is considered in a risk stage 

# In[44]:


risk_results = pd.DataFrame({
    "Actual_G3": y_test,
    "Predicted_G3": mid_rf_predictions
})

risk_results["Risk_Status"] = risk_results["Predicted_G3"].apply(
    lambda score: "At Risk" if score < 10 else "Not At Risk"
)

risk_results.head()

risk_results["Risk_Status"].value_counts()


# In[45]:


plt.figure(figsize=(6, 4))
sns.countplot(x="Risk_Status", data=risk_results)
plt.title("Predicted At-Risk Students")
plt.show()


# Students with predicted `G3` below 10 were classified as "At Risk", since 10 is commonly considered the passing threshold. This risk flag can help identify students who may need academic support.
# 
# However, this should be used only as a support tool and not as a final judgment about a student.

# **Feature Importance**  - Feature importance tells us which columns the Random Forest used most.

# In[46]:


preprocessor = mid_rf_model.named_steps["preprocessor"]
model = mid_rf_model.named_steps["model"]

numeric_features = X_mid.select_dtypes(include=["number"]).columns

categorical_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
    X_mid.select_dtypes(include=["object"]).columns
)

all_features = list(numeric_features) + list(categorical_features)

feature_importance = pd.DataFrame({
    "Feature": all_features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

feature_importance.head(15)


# In[47]:


plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(15)
)

plt.title("Top 15 Feature Importances - Mid-Year Random Forest")
plt.show()


# The feature importance results show which variables contributed most to the Random Forest model. `G1` and `G2` are expected to be among the most important features, confirming that previous academic performance is the strongest predictor of final grade.

# ## Conclusion
# 
# This project compared two regression approaches for predicting students' final grade `G3`.
# 
# The early-year model used demographic, social, and behavioral features without `G1` and `G2`. Its performance was limited, with the best early-year model achieving an R2 score of 0.31 and an MAE of 3.00.
# 
# The mid-year model included all available features, including `G1` and `G2`. This model performed much better, with the best model achieving an R2 score of 0.80 and an MAE of 1.21.
# 
# Overall, previous academic performance was the strongest predictor of final grade. Early-year features can provide some useful information, but predictions become much more reliable once previous grades are available.

# ## Limitations
# 
# The dataset is relatively small, so the results may not generalize to all students. Also, the model shows patterns in the data but does not prove that one factor directly causes another. The predictions should be used as a support tool, not as a final judgment about a student.

# ## Future Work
# 
# Future improvements could include cross-validation, hyperparameter tuning, feature selection, and testing the model on a larger dataset.
