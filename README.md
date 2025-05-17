# Predicting Credit Card Default Risk 🚀

**Authors:** Ganesh Sai Uttej Kayala, Michele Perina, Robert Mikkelson, Sharifur Rahman, Ranjith Kumar Saila

**Course:** MSBA 5314, University of Central Oklahoma

**Instructor:** Dr. Ho-Chang Chae

**Date:** December 12th, 2024

---

## Abstract

Understanding factors that influence credit card default rates is critical for financial institutions aiming to manage risk effectively while fostering inclusivity. This study focuses on assessing the impact of migrant worker status on credit card default rates, a demographic that remains largely unexplored in existing literature. Using a dataset of American Express customers, we analyzed variables such as demographics, financial behaviors, employment, and socio-economic attributes to uncover patterns and relationships. Advanced machine learning models, including decision trees, logistic regression, and neural networks, were employed to evaluate default likelihood, with and without credit-related variables. Migrant worker status emerged as a significant predictor, with migrant workers showing marginally higher odds of default. While credit-related factors like credit score and limit utilization remained dominant predictors, focusing on non-credit variables highlighted actionable opportunities for financial inclusivity. These findings suggest that tailored strategies addressing migrant workers’ unique financial circumstances can reduce default risks while opening avenues for expanding credit access responsibly.

---

## 📖 Table of Contents
* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Methodology](#methodology)
    * [Data Preprocessing](#data-preprocessing)
    * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    * [Statistical Analysis](#statistical-analysis)
    * [Modeling](#modeling)
* [Key Findings](#key-findings)
    * [Models with Credit Variables](#models-with-credit-variables)
    * [Models without Credit Variables (Non-Credit Variables)](#models-without-credit-variables-non-credit-variables)
* [Technologies Used](#technologies-used)
* [How to Use This Repository](#how-to-use-this-repository) (Optional)
* [Full Report](#full-report) (Optional)
* [Contributors](#contributors)

---

## 🌟 Project Overview

This project aims to predict credit card default risk using a unique dataset of American Express customers. Key objectives include:
1.  Investigating the predictive power of various demographic, financial, and behavioral attributes.
2.  Specifically assessing the impact of **migrant worker status** on default rates, a novel aspect in this domain.
3.  Employing a dual-model approach:
    * Models including traditional credit-related variables for maximum predictive accuracy.
    * Models excluding credit-related variables to highlight the influence of often-overlooked demographic and behavioral factors, particularly for individuals with limited or no credit history (e.g., migrant workers).
4.  Providing insights that can help financial institutions like American Express manage risk more effectively and promote financial inclusivity.

---

## 📊 Dataset

* **Source:** Kaggle, originally from the AmExpert 2021 – Code Lab competition (a hackathon by American Express and HackerEarth).
* **Description:** The dataset contains information on American Express customers, including demographics, employment details, financial behaviors, and credit-related attributes.
* **Initial Size:** 45,528 observations and 19 variables.
* **Final Size (after cleaning):** 29,475 observations.
* **Target Variable:** `Credit Card Default` (Binary: Yes/No).

Key variables analyzed include: Age, Gender, Owns Car, Owns House, Yearly Income, No. of days Employed, Occupation, Total Family Members, **Migrant Worker**, Yearly Debt Payment, Credit Limit, Credit Limit Used (%), Credit Score, Previous Defaults, Default in last 6 Months.

---

## ⚙️ Methodology

### Data Preprocessing
1.  **Error Handling:** Removed observations with data entry errors and inconsistencies (e.g., employment duration exceeding age, extreme outliers in income/credit limit). 16,051 rows removed due to age/employment inconsistency. Anomalous gender value "X" imputed with mode ("female").
2.  **Missing Value Imputation:**
    * Numeric Variables (e.g., `No. of Children`, `Yearly Debt Payments`, `Credit Score`): Mean imputation using SAS Miner's Impute Node.
    * Categorical Variables (e.g., `Migrant Worker`): Mode imputation.
3.  **Outlier Treatment:** Applied log transformation (using SAS Miner's Transform Node) to variables with skewness > 3 (e.g., `Previous Defaults`, `Credit Limit`, `Net Yearly Income`) to normalize distributions.
4.  **Variable Exclusion (due to multicollinearity or excessive predictive power):**
    * `Credit Limit` (high correlation with `Net Yearly Income`).
    * `No. of Children` (high correlation with `Total Family Members`).
    * `Name` and `Customer ID` (no predictive value or used as ID).
    * `Previous Defaults` and `Default in last 6 Months` (too strong predictors, causing complete separation and hindering insight).

### Exploratory Data Analysis (EDA)
* Generated summary statistics for interval and categorical variables.
* Analyzed frequency distributions.
* Investigated correlations between interval variables (Pearson Correlation Coefficients).
* **Feature Engineering:**
    * Converted `No. of days Employed` to `No. of Years Employed`.
    * Grouped `Occupation Type` into 'Professional' and 'Non-Professional' categories.

### Statistical Analysis
Performed various statistical tests to assess relationships between variables:
* **T-Tests:**
    * Mean Credit Score vs. Default.
    * Mean Credit Limit Used (%) vs. Default.
    * Mean Number of Years Employed vs. Default.
* **ANOVA Test:**
    * Occupation Type vs. Mean Credit Score.
* **Chi-Square Test:**
    * Credit Card Default vs. Migrant Worker status.

### Modeling
A dual-dataset approach was adopted:
1.  **Dataset 1:** Included credit-related variables.
2.  **Dataset 2:** Excluded credit-related variables.

For each dataset, 26 models were developed:
* **Addressing Class Imbalance:** Used a cost-sensitive approach (inverse class distribution weighting; minority class weighted 11.11) instead of undersampling to mitigate bias without losing data.
* **Data Partition:** 50% for training, 50% for validation.
* **Model Types:**
    * **Decision Trees (1):** Misclassification as subtree assessment.
    * **Random Forests (2):** One with 100 trees (default), one with 200 trees.
    * **Logistic Regression (11):** Various variable selection techniques (Stepwise, Variable Selection Node for R-squared, Partial Least Squares (PLS), Decision Tree-based selection, PCA-based selection, Variable Clustering, LARS, LASSO, Adaptive LASSO).
    * **Neural Networks (8):** Using similar variable selection techniques as logistic regression.
    * **Support Vector Machines (SVMs) (4):** Explored Linear, Polynomial, Radial Basis Function (RBF), and Sigmoid kernels.
* **Primary Evaluation Metric:** Accuracy.
* **Secondary Metrics:** Sensitivity, Specificity, Precision, F1 Score (with particular emphasis on Sensitivity for default prediction).

---

## 💡 Key Findings

### Models with Credit Variables
* **Champion Model:** Random Forest (Large - 200 trees) with **96% accuracy** and 96% specificity.
    * Matched by Stepwise Neural Network in accuracy (95%) but Random Forest had higher specificity.
* **Dominant Predictors:** `Credit Score` and `Credit Limit Used (%)` were consistently selected by all 26 models.
* **Odds Ratios (from PLS Regression):**
    * `Credit Score`: Odds of default decrease by 6% for each unit increase.
    * `Credit Limit Used (%)`: Odds of default increase by 11% for each 1% increase in usage.
    * `Migrant Worker Status`: Odds ratio ~1 (0.99), suggesting nearly identical default likelihood compared to non-migrant workers *when credit variables are present*.
* The Decision Tree model primarily split on `Credit Score` and `Credit Limit Used (%)`, offering limited new insights beyond these strong predictors.

### Models without Credit Variables (Non-Credit Variables)
This analysis aimed to uncover the impact of other variables when strong credit predictors are absent (e.g., for customers with no credit history).
* **Champion Model:** Logistic Regression (with PLS for variable selection) based on accuracy and sensitivity (Accuracy: 0.59, Sensitivity: 0.64).
* **Prominent Predictors (more frequently selected than in credit-variable models):**
    * `Owns Car` (most frequently used).
    * `No. of Years Employed`.
    * `Gender`.
    * `Net Yearly Income`.
    * `Occupation Type`.
    * `Migrant Worker` status (ranked among top five most utilized).
* **Odds Ratios (from PLS Regression):**
    * `Owns Car (No vs. Yes)`: Odds of default 1.42 times higher for those without a car.
    * `Gender (Male vs. Female)`: Males have 1.52 times higher odds of defaulting.
    * `Occupation Type (Non-Professional vs. Professional)`: Non-professionals have 1.16 times higher odds of defaulting.
    * `Migrant Worker (Yes vs. No)`: Migrant workers have 1.06 times higher odds of defaulting (a more distinct effect than when credit variables were included, which was 0.94).
    * `No. of Years Employed (Low-2 vs 7-High)`: Those with 0-2 years of employment have 1.55 times higher odds of defaulting compared to those employed for 7+ years.

### Additional Segmentations
* Segmenting by income (poor vs. rich) did not yield significant differences.
* Segmenting by occupation type (professional vs. non-professional) when using only non-credit variables showed slightly higher accuracy and other metrics for the "professional" group, suggesting a potential area for future research.

---

## 💻 Technologies Used
* **Primary Tool:** SAS Miner 15.2 (for data preprocessing, imputation, transformations, modeling, and evaluation).
* (Mention Python/R if used for any supplementary analysis or visualization).

---

## 🚀 How to Use This Repository (Optional)
*(If you are sharing code, SAS Miner process flows, or datasets)*
* **Prerequisites:** (e.g., SAS Miner 15.2 installed)
* **Setup:** (e.g., How to import the dataset, how to open the SAS Miner diagram)
* **Running the Analysis:** (e.g., Steps to execute the models)
* **Folder Structure:**
    * `/data`: Contains the dataset(s) used.
    * `/sas_miner_flows`: Contains SAS Miner diagrams or project files.
    * `/reports`: Contains the full project report or presentations.
    * `/scripts`: (If any Python/R scripts were used).

---

## 📄 Full Report (Optional)
* A link to the full PDF report can be provided here if you plan to host it (e.g., on Google Drive, Dropbox, or as a PDF in the repository).

---

## ✨ Contributors
* Ganesh Sai Uttej Kayala
* Michele Perina
* Robert Mikkelson
* Sharifur Rahman
* **Ranjith Kumar Saila** ([Your GitHub Profile Link](YOUR_GITHUB_PROFILE_LINK_HERE))

---
Next Steps for You:Create a README.md file in the root of your GitHub repository for this project.Copy and paste the content above into that file.Customize:Replace (YOUR_GITHUB_PROFILE_LINK_HERE) with the actual link to your GitHub profile.Fill in the "Technologies Used" if you used more than SAS Miner.Decide if you want to include the "How to Use This Repository" and "Full Report" sections. If so, provide the necessary details or links. If you are not sharing code or the full dataset directly, you might remove the "How to Use This Repository" section or modify it to explain what is available.If you have visualizations or key tables from your SAS Miner output (like the model comparison tables or variable importance charts), consider taking screenshots and embedding them in the README for better visual appeal. GitHub Markdown supports images: ![Alt text for image](path/to/image.png).Review and refine the language to ensure it accurately reflects your project and your personal style.This structure should give a very professional and comprehensive overview of your excellent final project on your GitHub profile!
