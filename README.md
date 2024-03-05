# 2024 Allianz Practitioner's Challenge - Group 3
# Prediction Models for Highly Zero-Inflated Claims Count

This GitHub repository contains all the Jupyter notebooks created for processing the dataset and implementing various methods for making predictions on highly zero-inflated insurance claims data. The repository features a total of six branches, including the main branch and five additional branches, each created by individual members for uploading their respective parts of the Jupyter notebook. The main branch serves as the central repository for all crucial final notebooks and the test environment.

### Introduction
The 2024 Allianz Practitioner's Challenge centered on predicting insurance claims data with a high frequency of zero claims (96% zeros). Our team tackled this by evaluating various statistical and machine learning models tailored for zero-inflated data along with some latest data scientist free tools like AutoML. The challenge aimed to get the best possible accuracy and explainability (Poisson Deviance Error, PDE) by incorporating innovative approaches.

### Challenge
The insurance claims data we dealt with was highly skewed towards zeroes (96%), and the remaining non-zero claims were rare and important cases. The key objective was to utilize techniques such as REBAGG (REsampled BAGGing for Imbalanced Regression) and Deep Imbalanced Regression to make better use of these rare and important cases, as predicting them accurately is crucial for insurance companies to determine appropriate policy pricing based on risk (number of claims).

### Exploratory Data Analysis (EDA)
From the EDA, we found that:
The 'Area' feature could be dropped as it can be inferred from the 'Region' feature.
The 'ClaimAmount' feature was dropped as it cannot be used to predict the number of claims.
The 'B12RN' feature had a higher claim frequency, as noted in a Kaggler's notebook.
We also performed feature engineering techniques like high-order feature transformations, categorization, interaction detection, and extraction.
Additionally, we utilized external data sources like traffic cases in each region from data.gouv.fr to enrich our dataset.

### Methodologies
Statistical Models
Zero-Inflated Poisson Model: Combines Poisson and binomial distributions to model excess zero counts separately.
Hurdle Poisson Model: Two-stage model using binary classification for zero counts, then zero-truncated Poisson for positive counts.
Machine Learning Models
CANN (Combined Actuarial Neural Network): Integrates a GLM with a neural network to capture non-linear relationships.
Boosting Models:
XGBoost: Highly efficient gradient boosting implementation.
CatBoost: Based on GBDT, optimized for categorical features with ordered boosting and symmetric trees.
LightGBM: Gradient boosting framework designed for distributed and high-dimensional data.
AutoML: Automated machine learning pipeline for data preprocessing, feature engineering, model selection and tuning.

Imbalanced Data Techniques
REBAGG: Resampled bagging technique for imbalanced regression tasks.
Deep Imbalanced Regression:
Label Distribution Smoothing (LDS): Smooths label density with kernel density estimation.
Feature Distribution Smoothing (FDS): Transfers feature statistics between target bins.

###Ensemble Modeling
A stacking ensemble model was created by combining predictions from multiple base models like XGBoost, LightGBM and their variants using a meta-learner like XGBoost or LightGBM.

###Evaluation Metrics
The utils notebook, located in the `srcs` folder, computes metrics like Mean Absolute Error (MAE), Mean Poisson Deviance, Poisson Deviance Error (PDE) - the winning criteria, training time and memory usage.

###Results
The stacking ensemble model achieved the best trade-off between accuracy (low MAE) and explainability (low PDE) by leveraging various techniques to handle the highly imbalanced claims data effectively.

### Winning Approach
Our winning approach involved a stacked ensemble model that combined the strengths of multiple base models, including XGBoost, LightGBM, and their variants with REBAGG and hyperparameter tuning. This ensemble model achieved the best trade-off between accuracy (low MAE) and explainability (low PDE), leveraging the power of various techniques to effectively handle the highly imbalanced claims data.

### Future Work
As insurance data evolves with the introduction of autonomous vehicles, our future work will focus on adapting our models to the changing landscape. We plan to explore how autonomous cars can possibly redefine insurance and how AI-driven vehicles will impact the prediction of insurance claims.

### Getting Started
To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-repo-url.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Navigate to the appropriate notebook and run the cells to see the results.

### Contributors
- [Contributor 1]
- [Contributor 2]
- [Contributor 3]
- [Contributor 4]
- [Contributor 5]

### Acknowledgments
We would like to express our gratitude to Allianz and the organizers of the 2024 Practitioner's Challenge for providing us with this exciting opportunity to work on a real-world problem and push the boundaries of our knowledge.


